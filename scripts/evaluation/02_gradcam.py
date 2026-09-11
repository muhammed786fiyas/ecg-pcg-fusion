"""Contribution 2: Grad-CAM on a trained fusion model, reported in Hz.

Hooks the last Conv2d of each branch - the final conv of CNNBranch for the
custom-CNN families, the 1x1 projection on layer4's 7x7 map for
cross_attn_resnet18 - computes a class activation map per modality, and
reports where the map puts its mass in real frequency bands.

Two outputs per fold:
  - FIGURES for a small, deliberate spread of segments: correctly classified
    normal, correctly classified abnormal, and misclassifications - the last of
    these matter most for the write up and are the easiest to quietly leave out.
  - a frequency PROFILE of every test segment. Twelve hand-picked panels are an
    illustration, not a measurement; the band shares the paper quotes come from
    all of a fold's test segments, then mean +/- std across folds.

The map is for the abnormal logit, as in the original day-1 analysis, so it
shows what pushes a segment towards "abnormal". A map that ReLU zeroes out
entirely has no mass to place; such segments are counted and excluded from the
band averages rather than averaged in as zeros.

Band shares are NOT read raw. The image rows are evenly spaced in wavelet SCALE,
and frequency goes as 1/scale, so the axis is hyperbolic in Hz: 62.5% of the ECG
rows lie between 4 and 10 Hz. A map that prefers nothing - a uniform one - would
already put 62.5% of its mass there. Each share is reported beside two
baselines: the uniform-map share (pure axis geometry) and the share of the
scalogram's own intensity (does the map just follow bright pixels?). Only a
departure from those says anything about what the model prefers. Day 1's
"93.6% of ECG mass below 10 Hz" was read against zero, not against 62.5%.

Grad-CAM is hand-rolled rather than taken from the `grad-cam` package: it is a
forward hook, a backward hook and four lines of arithmetic, and inlining it
keeps this script runnable inside a Kaggle kernel with no extra dependency.
Batching is exact here: in eval mode each segment's logit depends only on its
own input, so the gradient of the summed logits with respect to a segment's
activations is that segment's own gradient. tests/test_gradcam.py pins this.

The interpretation itself goes in docs/logs/tasks/4-interpretability.md, and it
is to be written honestly: if the ECG map does not concentrate on the QRS
time-frequency region and the PCG map on the S1/S2 bursts, that is the result.
"""

import argparse
import importlib.util
import json
import os

import matplotlib
import numpy as np
import pandas as pd
import pywt
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

UINT8_MAX = 255.0
IMAGE_SIZE = 224
OVERLAY_ALPHA = 0.45
CAM_EPSILON = 1e-8
CAM_BATCH_SIZE = 32
MODALITIES = ["ecg", "pcg"]
SUBSETS = ["all", "normal", "abnormal"]

# Named by what the band means for the heart, not by equal division of the axis.
# QRS energy is classically 5-40 Hz; S1/S2 heart sounds sit around 20-150 Hz.
FREQUENCY_BANDS = [
    ("0_10hz", 0.0, 10.0),
    ("10_25hz", 10.0, 25.0),
    ("25_50hz", 25.0, 50.0),
    ("50_plus_hz", 50.0, 1e9),
]
BAND_METRICS = ["mass_0_10hz", "mass_10_25hz", "mass_25_50hz", "mass_50_plus_hz"]
SUMMARY_METRICS = [
    "mass_0_10hz", "mass_10_25hz", "mass_25_50hz", "mass_50_plus_hz",
    "median_frequency_hz", "peak_frequency_hz", "time_concentration",
]


def load_params(params_path):
    with open(params_path) as handle:
        return yaml.safe_load(handle)


def load_training_module(family):
    path = os.path.join("scripts", "modeling", family, "01_train.py")
    if not os.path.exists(path):
        raise SystemExit(f"QC FAIL: {path} not found")
    spec = importlib.util.spec_from_file_location("train_" + family, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def last_conv_layer(branch):
    """The last Conv2d inside a branch - where Grad-CAM hooks.

    For CNNBranch that is the final conv of its feature stack (a 14x14 map); for
    the ResNet branch it is the 1x1 projection on layer4 (a 7x7 map).
    """
    convs = [layer for layer in branch.modules() if isinstance(layer, torch.nn.Conv2d)]
    if len(convs) == 0:
        raise SystemExit("QC FAIL: no Conv2d found in the branch")
    return convs[-1]


class GradCam:
    """Forward and backward hooks on one conv layer.

    activations: the layer's output for the current forward pass.
    gradients:   d(target logit) / d(that output).
    The map is ReLU over the channel-weighted sum, weights being the spatially
    averaged gradients - Selvaraju et al., ICCV 2017.
    """

    def __init__(self, layer):
        self.activations = None
        self.gradients = None
        self.forward_handle = layer.register_forward_hook(self.save_activations)
        self.backward_handle = layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module, layer_input, layer_output):
        self.activations = layer_output.detach()

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def compute_all(self):
        """One map per example in the batch, each scaled to [0, 1] on its own."""
        if self.activations is None or self.gradients is None:
            raise SystemExit("QC FAIL: Grad-CAM hooks captured nothing - did backward() run?")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cams = torch.relu((weights * self.activations).sum(dim=1)).cpu().numpy()
        normalised = []
        for cam in cams:
            cam = cam - cam.min()
            peak = cam.max()
            if peak > CAM_EPSILON:
                cam = cam / peak
            normalised.append(cam)
        return normalised

    def compute(self):
        return self.compute_all()[0]

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


def upsample_cam(cam, size):
    """Bilinear upsample of the small CAM to the image grid."""
    tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)
    resized = torch.nn.functional.interpolate(
        tensor, size=(size, size), mode="bilinear", align_corners=False
    )
    return resized[0, 0].numpy()


def pick_segments(predictions, threshold, n_wanted):
    """A spread over correct-normal, correct-abnormal and misclassified.

    Misclassifications are taken deliberately rather than by luck: they are the
    most informative panels and the easiest to omit by accident.
    """
    frame = predictions.copy()
    frame["prediction"] = (frame["prob"] >= threshold).astype(int)
    frame["correct"] = frame["prediction"] == frame["label"]

    groups = {
        "correct_normal": frame[(frame["correct"]) & (frame["label"] == 0)].sort_values("prob"),
        "correct_abnormal": frame[(frame["correct"]) & (frame["label"] == 1)].sort_values("prob", ascending=False),
        "false_positive": frame[(~frame["correct"]) & (frame["label"] == 0)].sort_values("prob", ascending=False),
        "false_negative": frame[(~frame["correct"]) & (frame["label"] == 1)].sort_values("prob"),
    }

    per_group = max(1, n_wanted // len(groups))
    chosen = []
    for name, group in groups.items():
        taken = group.head(per_group)
        for row in taken.itertuples(index=False):
            chosen.append({"category": name, "row": row})
        print(f"  {name}: {len(group)} available, took {len(taken)}")
    return chosen[:n_wanted]


def render_panel(ecg_image, pcg_image, ecg_cam, pcg_cam, title, out_path):
    figure, axes = plt.subplots(2, 2, figsize=(9, 8))

    axes[0][0].imshow(ecg_image, cmap="jet", aspect="auto", origin="lower")
    axes[0][0].set_title("ECG scalogram")
    axes[0][1].imshow(ecg_image, cmap="gray", aspect="auto", origin="lower")
    axes[0][1].imshow(ecg_cam, cmap="jet", alpha=OVERLAY_ALPHA, aspect="auto", origin="lower")
    axes[0][1].set_title("ECG Grad-CAM")

    axes[1][0].imshow(pcg_image, cmap="jet", aspect="auto", origin="lower")
    axes[1][0].set_title("PCG scalogram")
    axes[1][1].imshow(pcg_image, cmap="gray", aspect="auto", origin="lower")
    axes[1][1].imshow(pcg_cam, cmap="jet", alpha=OVERLAY_ALPHA, aspect="auto", origin="lower")
    axes[1][1].set_title("PCG Grad-CAM")

    for row in axes:
        for axis in row:
            axis.set_xlabel("time (samples, 3 s window)")
            axis.set_ylabel("scale (low freq at top)")

    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(out_path, dpi=130)
    plt.close(figure)


def row_frequencies_hz(scales, wavelet, fs, n_rows):
    """The frequency each image row stands for.

    The scalogram was resized from len(scales) rows to n_rows, so row r came
    from this position on the original scale axis.
    """
    positions = np.linspace(0, len(scales) - 1, n_rows)
    row_scales = np.interp(positions, np.arange(len(scales)), np.asarray(scales, dtype=float))
    return pywt.scale2frequency(wavelet, row_scales) * fs


def cam_mass_profile(cam, row_frequencies):
    """Where the map puts its mass, reported in actual Hz.

    An earlier version binned rows into "low/mid/high thirds" by ROW INDEX and
    named them accordingly. That inverted the physiology: row 0 is the SMALLEST
    scale, which is the HIGHEST frequency, so the field called "low" held the
    high-frequency mass. Reporting it would have claimed the opposite of what the
    model attends to.

    Rows are mapped back to their scale, and the scale to Hz, so the bands are
    named by the quantity a cardiologist would ask about.
    """
    row_mass = cam.mean(axis=1)
    total = float(row_mass.sum()) + CAM_EPSILON

    def band_share(low_hz, high_hz):
        inside = (row_frequencies >= low_hz) & (row_frequencies < high_hz)
        return float(row_mass[inside].sum() / total)

    time_mass = cam.mean(axis=0)
    profile = {
        "cam_empty": bool(float(cam.max()) <= CAM_EPSILON),
        "peak_frequency_hz": float(row_frequencies[int(np.argmax(row_mass))]),
        "median_frequency_hz": float(
            row_frequencies[int(np.argmin(np.abs(np.cumsum(row_mass) - total / 2.0)))]
        ),
        "time_peak_position": float(np.argmax(time_mass)) / float(len(time_mass)),
        "time_concentration": float(time_mass.max() / (time_mass.mean() + CAM_EPSILON)),
        "band_min_hz": float(row_frequencies.min()),
        "band_max_hz": float(row_frequencies.max()),
    }
    # Bands chosen for what they mean clinically, not for equal width.
    for name, low_hz, high_hz in FREQUENCY_BANDS:
        profile["mass_" + name] = band_share(low_hz, high_hz)
    return profile


def load_images(ecg_memmap, pcg_memmap, memmap_rows):
    ecg = np.asarray(ecg_memmap[memmap_rows], dtype=np.float32) / UINT8_MAX
    pcg = np.asarray(pcg_memmap[memmap_rows], dtype=np.float32) / UINT8_MAX
    return ecg, pcg, torch.from_numpy(ecg).unsqueeze(1), torch.from_numpy(pcg).unsqueeze(1)


def batch_cams(model, hooks, ecg_tensor, pcg_tensor):
    """Upsampled ECG and PCG maps plus probabilities for a batch of segments."""
    model.zero_grad(set_to_none=True)
    logits = model(ecg_tensor, pcg_tensor)
    logits.sum().backward()
    ecg_cams = [upsample_cam(cam, IMAGE_SIZE) for cam in hooks["ecg"].compute_all()]
    pcg_cams = [upsample_cam(cam, IMAGE_SIZE) for cam in hooks["pcg"].compute_all()]
    return ecg_cams, pcg_cams, torch.sigmoid(logits.detach()).numpy()


def profile_segments(model, hooks, frame, row_of, memmaps, frequencies):
    """Frequency profile of the CAMs of every segment in the frame."""
    records = []
    for start in range(0, len(frame), CAM_BATCH_SIZE):
        chunk = frame.iloc[start:start + CAM_BATCH_SIZE]
        memmap_rows = [int(row_of[variant_id]) for variant_id in chunk["segment_variant_id"]]
        ecg_images, pcg_images, ecg_tensor, pcg_tensor = load_images(memmaps["ecg"], memmaps["pcg"], memmap_rows)
        ecg_cams, pcg_cams, probabilities = batch_cams(model, hooks, ecg_tensor, pcg_tensor)
        for index, row in enumerate(chunk.itertuples(index=False)):
            record = {
                "segment_variant_id": row.segment_variant_id,
                "record_id": row.record_id,
                "label": int(row.label),
                "prob": float(probabilities[index]),
            }
            for name, value in cam_mass_profile(ecg_cams[index], frequencies["ecg"]).items():
                record["ecg_" + name] = value
            for name, value in cam_mass_profile(pcg_cams[index], frequencies["pcg"]).items():
                record["pcg_" + name] = value
            # The scalogram-energy baseline: the same profile of the image itself.
            ecg_energy = cam_mass_profile(ecg_images[index], frequencies["ecg"])
            pcg_energy = cam_mass_profile(pcg_images[index], frequencies["pcg"])
            for name in BAND_METRICS:
                record["ecg_image_" + name] = ecg_energy[name]
                record["pcg_image_" + name] = pcg_energy[name]
            records.append(record)
    return pd.DataFrame(records)


def summarise_fold(profiles, fold_tag):
    """Mean band shares per modality and subset, empty maps excluded."""
    rows = []
    for modality in MODALITIES:
        for subset in SUBSETS:
            chosen = profiles
            if subset == "normal":
                chosen = profiles[profiles["label"] == 0]
            if subset == "abnormal":
                chosen = profiles[profiles["label"] == 1]
            usable = chosen[~chosen[modality + "_cam_empty"]]
            row = {
                "fold": fold_tag,
                "modality": modality,
                "subset": subset,
                "segments": len(chosen),
                "empty_maps": int(chosen[modality + "_cam_empty"].sum()),
            }
            for metric in SUMMARY_METRICS:
                row[metric] = float(usable[modality + "_" + metric].mean())
            for metric in BAND_METRICS:
                row["image_" + metric] = float(usable[modality + "_image_" + metric].mean())
            rows.append(row)
    return rows


def render_examples(model, hooks, chosen, row_of, memmaps, frequencies, out_dir):
    """Figures for the hand-picked spread of segments."""
    records = []
    for entry in chosen:
        row = entry["row"]
        variant_id = row.segment_variant_id
        ecg_images, pcg_images, ecg_tensor, pcg_tensor = load_images(
            memmaps["ecg"], memmaps["pcg"], [int(row_of[variant_id])]
        )
        ecg_cams, pcg_cams, probabilities = batch_cams(model, hooks, ecg_tensor, pcg_tensor)
        probability = float(probabilities[0])
        label_name = "abnormal" if int(row.label) == 1 else "normal"
        title = (
            f"{variant_id} | record {row.record_id} | true={label_name} "
            f"| p(abnormal)={round(probability, 3)} | {entry['category']}"
        )
        out_path = os.path.join(out_dir, entry["category"] + "_" + variant_id + ".png")
        render_panel(ecg_images[0], pcg_images[0], ecg_cams[0], pcg_cams[0], title, out_path)

        record = {
            "segment_variant_id": variant_id,
            "record_id": row.record_id,
            "label": int(row.label),
            "prob": probability,
            "category": entry["category"],
            "figure": os.path.basename(out_path),
        }
        for name, value in cam_mass_profile(ecg_cams[0], frequencies["ecg"]).items():
            record["ecg_" + name] = value
        for name, value in cam_mass_profile(pcg_cams[0], frequencies["pcg"]).items():
            record["pcg_" + name] = value
        records.append(record)
    return records


def run_fold(args, params, module, fold_tag, out_dir, frequencies):
    threshold = params["evaluation"]["decision_threshold"]
    n_segments = params["evaluation"]["gradcam_n_segments"]
    print(f"--- {fold_tag} ---")

    model = module.build_model(params)
    checkpoint = os.path.join(
        args.model_root, args.family, args.scalogram_config,
        f"{args.scalogram_config}_{fold_tag}_best.pth",
    )
    if not os.path.exists(checkpoint):
        raise SystemExit(f"QC FAIL: checkpoint not found: {checkpoint}. Train {args.family} first.")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"loaded {checkpoint} (val_auc={round(float(state.get('val_auc', float('nan'))), 4)})")

    predictions_path = os.path.join(
        args.report_root, args.family, args.scalogram_config, fold_tag, "test_predictions.csv"
    )
    if not os.path.exists(predictions_path):
        raise SystemExit(f"QC FAIL: {predictions_path} not found")
    predictions = pd.read_csv(predictions_path)

    scalogram_dir = os.path.join(args.scalogram_root, args.scalogram_config)
    row_index = pd.read_csv(os.path.join(scalogram_dir, "row_index.csv"))
    row_of = dict(zip(row_index["segment_variant_id"], row_index["row_index"], strict=True))
    memmaps = {
        "ecg": np.load(os.path.join(scalogram_dir, "ecg.uint8.npy"), mmap_mode="r"),
        "pcg": np.load(os.path.join(scalogram_dir, "pcg.uint8.npy"), mmap_mode="r"),
    }

    os.makedirs(out_dir, exist_ok=True)
    hooks = {
        "ecg": GradCam(last_conv_layer(model.ecg_branch)),
        "pcg": GradCam(last_conv_layer(model.pcg_branch)),
    }

    chosen = pick_segments(predictions, threshold, n_segments)
    examples = render_examples(model, hooks, chosen, row_of, memmaps, frequencies, out_dir)
    pd.DataFrame(examples).to_csv(os.path.join(out_dir, "gradcam_summary.csv"), index=False)
    with open(os.path.join(out_dir, "gradcam_summary.json"), "w") as handle:
        json.dump(examples, handle, indent=2)
    print(f"  wrote {len(examples)} example figures")

    profiles = profile_segments(model, hooks, predictions, row_of, memmaps, frequencies)
    for hook in hooks.values():
        hook.close()

    # The profile recomputes each probability; it must agree with the saved
    # test predictions or the maps are not of the model that was evaluated.
    drift = float(np.max(np.abs(profiles["prob"].to_numpy() - predictions["prob"].to_numpy())))
    if drift > 0.05:
        raise SystemExit(f"QC FAIL: recomputed probabilities drift by {round(drift, 4)} from the saved ones")
    profiles.to_csv(os.path.join(out_dir, "all_segments.csv"), index=False)

    rows = summarise_fold(profiles, fold_tag)
    for row in rows:
        if row["subset"] == "all":
            print(
                f"  {row['modality']}: {row['segments']} segments ({row['empty_maps']} empty maps) | "
                f"0-10 Hz {round(row['mass_0_10hz'], 3)}, 10-25 Hz {round(row['mass_10_25hz'], 3)}, "
                f"25-50 Hz {round(row['mass_25_50hz'], 3)}, 50+ Hz {round(row['mass_50_plus_hz'], 3)} | "
                f"median {round(row['median_frequency_hz'], 1)} Hz, drift {round(drift, 5)}"
            )
    return rows


def ratio_text(numerator, denominator):
    if denominator <= CAM_EPSILON:
        return "--"
    return f"{numerator / denominator:.2f}"


def baseline_table(frame, uniform):
    """Each band's Grad-CAM share against the uniform-map and image-energy shares."""
    rows = []
    for modality in MODALITIES:
        chosen = frame[(frame["modality"] == modality) & (frame["subset"] == "all")]
        for metric in BAND_METRICS:
            cam_share = float(chosen[metric].mean())
            energy_share = float(chosen["image_" + metric].mean())
            uniform_share = float(uniform[modality][metric])
            rows.append(
                {
                    "modality": modality.upper(),
                    "band": metric.replace("mass_", ""),
                    "Grad-CAM share": f"{cam_share:.3f} ± {chosen[metric].std(ddof=0):.3f}",
                    "uniform-map share": f"{uniform_share:.3f}",
                    "scalogram-energy share": f"{energy_share:.3f} ± {chosen['image_' + metric].std(ddof=0):.3f}",
                    "CAM / uniform": ratio_text(cam_share, uniform_share),
                    "CAM / energy": ratio_text(cam_share, energy_share),
                }
            )
    return pd.DataFrame(rows)


def write_cross_fold(fold_rows, family, out_root, uniform):
    frame = pd.DataFrame(fold_rows)
    frame.to_csv(os.path.join(out_root, "gradcam_per_fold.csv"), index=False)
    baselines = baseline_table(frame, uniform)
    baselines.to_csv(os.path.join(out_root, "gradcam_vs_baselines.csv"), index=False)

    table_rows = []
    for modality in MODALITIES:
        for subset in SUBSETS:
            chosen = frame[(frame["modality"] == modality) & (frame["subset"] == subset)]
            row = {"modality": modality.upper(), "subset": subset}
            for metric in SUMMARY_METRICS:
                row[metric] = f"{chosen[metric].mean():.3f} ± {chosen[metric].std(ddof=0):.3f}"
            row["empty maps"] = f"{int(chosen['empty_maps'].sum())}/{int(chosen['segments'].sum())}"
            table_rows.append(row)
    table = pd.DataFrame(table_rows)

    with open(os.path.join(out_root, "gradcam_cross_fold.md"), "w", encoding="utf-8") as handle:
        handle.write(f"# Grad-CAM frequency profile — {family}, all CV folds\n\n")
        handle.write(
            "Share of each modality's Grad-CAM mass by frequency band, over EVERY test "
            "segment of each fold (maps for the abnormal logit), then mean ± std across "
            "the folds. Maps that ReLU zeroes out entirely are excluded from the averages "
            "and counted in the last column. QRS energy is classically 5-40 Hz; S1/S2 "
            "heart sounds ~20-150 Hz. The ECG transform covers 4-100 Hz and the PCG "
            "transform 12.5-232 Hz, so bands outside those ranges are empty by "
            "construction.\n\n"
        )
        handle.write(table.to_markdown(index=False))
        handle.write("\n\n## Against the baselines\n\n")
        handle.write(
            "The image rows are evenly spaced in wavelet scale, so the frequency axis is "
            "hyperbolic: a UNIFORM map already puts most of its mass in the lowest band. "
            "The uniform-map share is that pure geometry; the scalogram-energy share is "
            "where the image's own intensity sits. A ratio near 1 means the map puts no "
            "more mass in a band than geometry or brightness alone would. Bands outside "
            "a transform's range show '--'.\n\n"
        )
        handle.write(baselines.to_markdown(index=False))
        handle.write("\n")
    print("")
    print(table[table["subset"] == "all"].to_string(index=False))
    print("")
    print(baselines.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM on a trained fusion model, in Hz")
    parser.add_argument("--family", default="cross_attn_resnet18")
    parser.add_argument("--scalogram-config", default="default")
    parser.add_argument("--fold-tag", default="all", help="a fold tag such as cv_fold0, or 'all'")
    parser.add_argument("--scalogram-root", default="data/processed/scalograms")
    parser.add_argument("--model-root", default="models")
    parser.add_argument("--report-root", default="reports")
    parser.add_argument("--output-dir", default="", help="defaults to reports/gradcam/<family>")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    torch.set_num_threads(os.cpu_count())

    # Needed to map CAM rows back to real frequencies, so the physiological
    # claim is stated in Hz rather than in image-row position.
    feature_params = params["features"]["scalogram"]
    fs = params["data"]["convert"]["expected_fs"]
    ecg_scales = np.arange(feature_params["ecg_scale_start"], feature_params["ecg_scale_stop"])
    pcg_scales = np.arange(feature_params["pcg_scale_start"], feature_params["pcg_scale_stop"])
    frequencies = {
        "ecg": row_frequencies_hz(ecg_scales, feature_params["ecg_wavelet"], fs, IMAGE_SIZE),
        "pcg": row_frequencies_hz(pcg_scales, feature_params["pcg_wavelet"], fs, IMAGE_SIZE),
    }

    fold_tags = [args.fold_tag]
    if args.fold_tag == "all":
        fold_tags = ["cv_fold" + str(i) for i in range(params["evaluation"]["cv_folds"])]
    out_root = args.output_dir
    if out_root == "":
        out_root = os.path.join("reports", "gradcam", args.family)

    print("=== 02_gradcam ===")
    print(f"family={args.family} config={args.scalogram_config} folds={fold_tags}")

    module = load_training_module(args.family)
    fold_rows = []
    for fold_tag in fold_tags:
        out_dir = os.path.join(out_root, fold_tag)
        fold_rows.extend(run_fold(args, params, module, fold_tag, out_dir, frequencies))

    # What a map that prefers nothing would score: pure axis geometry.
    uniform = {}
    for modality in MODALITIES:
        uniform[modality] = cam_mass_profile(np.ones((IMAGE_SIZE, IMAGE_SIZE)), frequencies[modality])

    os.makedirs(out_root, exist_ok=True)
    write_cross_fold(fold_rows, args.family, out_root, uniform)
    print("")
    print("QRS energy is classically 5-40 Hz; S1/S2 heart sounds ~20-150 Hz.")
    print("Interpret and write up honestly in docs/logs/tasks/4-interpretability.md")
    print(f"wrote {out_root}")


if __name__ == "__main__":
    main()
