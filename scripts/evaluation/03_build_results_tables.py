"""Render the paper's results tables from the MLflow experiment.

Three tables, deliberately kept separate:

  ablation_table       the main model comparison
  wavelet_table        Contribution 1, dual_cnn across wavelet configs
  split_protocol_table Contribution 4, the NEGATIVE CONTROL arms

The split-protocol table is written to its own file with its own caption, and
every row in it is labelled a negative control. It must never be readable as a
result of the method - that is the whole reason the experiment exists.

CV rows carry mean +/- std across folds, never the best fold.
"""

import argparse
import os

import mlflow
import numpy as np
import pandas as pd
import pywt
import yaml
from dotenv import load_dotenv

MAIN_FAMILY_ORDER = [
    ("ecg_only", "ECG-only CNN", "--", "Custom 4-block"),
    ("pcg_only", "PCG-only CNN", "--", "Custom 4-block"),
    ("dual_cnn", "Dual-branch concat", "Concatenation", "Custom 4-block"),
    ("warm_start_fusion", "Dual-branch + warm-start", "Concatenation", "Custom 4-block, unimodal init"),
    ("cbam_fusion", "Dual-branch + CBAM", "Channel+spatial attn", "Custom 4-block"),
    ("cross_attn_fusion", "**Dual-branch + cross-attn**", "Cross-modal attention", "Custom 4-block"),
    ("cross_attn_resnet18", "ResNet-18 + cross-attn", "Cross-modal attention", "Pretrained ResNet-18"),
]

NEGATIVE_CONTROL_CAPTION = (
    "**NEGATIVE CONTROL - not a result of the method.** Arms B and C deliberately "
    "violate patient-level separation in order to measure how much apparent "
    "performance a leaky split manufactures on this dataset. Only arm A is a valid "
    "estimate of performance. Architecture, data, seed and schedule are identical "
    "across all three arms; the only thing that differs is the partitioning."
)


def fetch_runs(tracking_uri, experiment_name):
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise SystemExit(f"QC FAIL: MLflow experiment '{experiment_name}' not found at {tracking_uri}")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], max_results=50000)
    if len(runs) == 0:
        raise SystemExit("QC FAIL: the MLflow experiment contains no runs")
    return runs


def child_runs(runs):
    """Per-fold runs only. Parents hold the aggregates and would double-count."""
    if "params.fold" not in runs.columns:
        return runs.iloc[0:0]
    return runs[runs["params.fold"].notna()].copy()


def format_mean_std(values):
    clean = [value for value in values if value is not None and not np.isnan(value)]
    if len(clean) == 0:
        return "--"
    if len(clean) == 1:
        return str(round(float(clean[0]), 4))
    return f"{round(float(np.mean(clean)), 4)} ± {round(float(np.std(clean)), 4)}"


def collect(runs, family_label, metric):
    column = "metrics." + metric
    if column not in runs.columns:
        return []
    subset = runs[runs["params.family_label"] == family_label] if "params.family_label" in runs.columns else runs.iloc[0:0]
    return subset[column].tolist()


def family_labels(runs):
    if "params.family_label" not in runs.columns:
        return []
    return sorted(runs["params.family_label"].dropna().unique())


def build_main_table(runs):
    rows = []
    for family, display, fusion, backbone in MAIN_FAMILY_ORDER:
        seg = collect(runs, family, "seg_auc")
        patient = collect(runs, family, "patient_auc")
        seg_f1 = collect(runs, family, "seg_f1")
        patient_acc = collect(runs, family, "patient_accuracy")
        patient_sens = collect(runs, family, "patient_sensitivity")
        patient_spec = collect(runs, family, "patient_specificity")
        rows.append(
            {
                "Model": display,
                "Fusion": fusion,
                "Backbone": backbone,
                "Folds": len([v for v in seg if v is not None and not np.isnan(v)]),
                "Seg AUC": format_mean_std(seg),
                "Seg F1": format_mean_std(seg_f1),
                "Patient AUC": format_mean_std(patient),
                "Patient Acc": format_mean_std(patient_acc),
                "Patient Sens": format_mean_std(patient_sens),
                "Patient Spec": format_mean_std(patient_spec),
            }
        )
    return pd.DataFrame(rows)


def measured_band(wavelet, scale_start, scale_stop, fs):
    """The frequency band a wavelet actually covers at the given fixed scales.

    This is reported per row because holding the SCALES fixed across wavelets
    does NOT hold the frequency BAND fixed: every wavelet has its own centre
    frequency, so the same scales map to different Hz. At fs=2000 with the ECG
    scales used here, cmor1.5-1.0 covers 4-100 Hz while mexh covers 1-25 Hz - a
    3.2x difference. Any wavelet comparison at fixed scales is therefore partly
    a frequency-band comparison, and the table has to say so.
    """
    scales = np.arange(scale_start, scale_stop)
    frequencies = pywt.scale2frequency(wavelet, scales) * fs
    return f"{round(float(frequencies.min()), 1)}-{round(float(frequencies.max()), 1)}"


def load_scale_params(params_path):
    with open(params_path) as handle:
        params = yaml.safe_load(handle)
    return params["features"]["scalogram"]


def build_wavelet_table(runs, scale_params, fs):
    """Contribution 1. dual_cnn across scalogram configs, so the only thing that
    varies is the input representation."""
    if "params.scalogram_config" not in runs.columns:
        return pd.DataFrame()
    subset = runs[runs["params.family_label"] == "dual_cnn"] if "params.family_label" in runs.columns else runs.iloc[0:0]
    if len(subset) == 0:
        return pd.DataFrame()

    rows = []
    for config in sorted(subset["params.scalogram_config"].dropna().unique()):
        config_runs = subset[subset["params.scalogram_config"] == config]
        ecg_wavelet = config_runs["params.ecg_wavelet"].iloc[0] if "params.ecg_wavelet" in config_runs.columns else "?"
        pcg_wavelet = config_runs["params.pcg_wavelet"].iloc[0] if "params.pcg_wavelet" in config_runs.columns else "?"
        note = "default config, = the dual_cnn result" if config == "default" else ""
        rows.append(
            {
                "Config": config,
                "ECG wavelet": ecg_wavelet,
                "ECG band (Hz)": measured_band(
                    ecg_wavelet, scale_params["ecg_scale_start"], scale_params["ecg_scale_stop"], fs),
                "PCG wavelet": pcg_wavelet,
                "PCG band (Hz)": measured_band(
                    pcg_wavelet, scale_params["pcg_scale_start"], scale_params["pcg_scale_stop"], fs),
                "Folds": len(config_runs),
                "Seg AUC": format_mean_std(config_runs["metrics.seg_auc"].tolist() if "metrics.seg_auc" in config_runs.columns else []),
                "Patient AUC": format_mean_std(config_runs["metrics.patient_auc"].tolist() if "metrics.patient_auc" in config_runs.columns else []),
                "Note": note,
            }
        )
    return pd.DataFrame(rows)


def build_split_protocol_table(runs):
    """Contribution 4. Every row here is a negative control except arm A."""
    arms = [
        ("dual_cnn", "A - correct (record-level)", "no", "Valid estimate"),
        ("dual_cnn_leaky_val", "B - leaky validation", "YES", "NEGATIVE CONTROL"),
        ("dual_cnn_fully_leaky", "C - fully leaky", "YES", "NEGATIVE CONTROL"),
    ]
    rows = []
    baseline_seg = None
    for label, display, leaked, status in arms:
        seg = collect(runs, label, "seg_auc")
        patient = collect(runs, label, "patient_auc")
        clean_seg = [v for v in seg if v is not None and not np.isnan(v)]
        mean_seg = float(np.mean(clean_seg)) if len(clean_seg) else None
        if label == "dual_cnn" and mean_seg is not None:
            baseline_seg = mean_seg

        if mean_seg is not None and baseline_seg is not None and label != "dual_cnn":
            delta = round(mean_seg - baseline_seg, 4)
            delta_text = ("+" if delta >= 0 else "") + str(delta)
        elif label == "dual_cnn":
            delta_text = "0 (reference)"
        else:
            delta_text = "--"

        rows.append(
            {
                "Arm": display,
                "Leaks?": leaked,
                "Status": status,
                "Folds": len(clean_seg),
                "Seg AUC": format_mean_std(seg),
                "Patient AUC": format_mean_std(patient),
                "Seg AUC delta vs A": delta_text,
            }
        )
    return pd.DataFrame(rows)


def write_table(frame, base_path, title, caption):
    if len(frame) == 0:
        print(f"  {title}: no runs yet, skipping")
        return
    frame.to_csv(base_path + ".csv", index=False)
    with open(base_path + ".md", "w", encoding="utf-8") as handle:
        handle.write("# " + title + "\n\n")
        if caption:
            handle.write(caption + "\n\n")
        handle.write(frame.to_markdown(index=False))
        handle.write("\n")
    print(f"  wrote {base_path}.md and .csv ({len(frame)} rows)")


def main():
    parser = argparse.ArgumentParser(description="build the paper's results tables from MLflow")
    parser.add_argument("--output-dir", default="reports/figures")
    parser.add_argument("--tracking-uri", default="")
    parser.add_argument("--experiment", default="")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--fs", type=int, default=2000)
    args = parser.parse_args()

    load_dotenv()
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking_uri = args.tracking_uri if args.tracking_uri else os.environ.get("MLFLOW_TRACKING_URI", "file:./models/mlflow_tracking")
    experiment_name = args.experiment if args.experiment else os.environ.get("MLFLOW_EXPERIMENT_NAME", "ecg-pcg-fusion")

    print("=== 03_build_results_tables ===")
    print(f"tracking_uri={tracking_uri} experiment={experiment_name}")

    runs = fetch_runs(tracking_uri, experiment_name)
    folds = child_runs(runs)
    print(f"{len(runs)} runs total, {len(folds)} per-fold runs")
    print(f"family labels present: {family_labels(folds)}")

    os.makedirs(args.output_dir, exist_ok=True)

    write_table(
        build_main_table(folds),
        os.path.join(args.output_dir, "ablation_table"),
        "Main ablation - ECG-PCG fusion on PhysioNet/CinC 2016 Training-A",
        "Record-level stratified 5-fold cross-validation. Values are mean ± std "
        "across folds, never the best fold. Patient-level metrics use mean-probability "
        "aggregation over each record's segments.",
    )

    write_table(
        build_wavelet_table(folds, load_scale_params(args.params), args.fs),
        os.path.join(args.output_dir, "wavelet_table"),
        "Contribution 1 - CWT mother wavelet sensitivity",
        "Architecture is dual_cnn throughout, so the only thing that varies is the "
        "input representation. `gaus4` replaces the originally planned `db4`, which "
        "pywt.cwt cannot use because db4 is a discrete orthogonal wavelet - see "
        "docs/logs/tasks/2-features.md.\n\n"
        "**Important caveat, stated plainly.** The scale ranges are held fixed "
        "across wavelets, but that does NOT hold the frequency band fixed: each "
        "wavelet has its own centre frequency, so the same scales map to different "
        "Hz. The measured band is given per row, and it varies by up to 3.2x "
        "(cmor1.5-1.0 covers 4-100 Hz on ECG where mexh covers 1-25 Hz). This "
        "comparison is therefore of *wavelets at fixed scales*, which confounds "
        "wavelet shape with frequency coverage - it is NOT a frequency-matched "
        "comparison of wavelet shape. A matched study would need per-wavelet "
        "scales, which is a larger experiment; note it as a limitation and as "
        "future work rather than claiming the shape effect in isolation.",
    )

    write_table(
        build_split_protocol_table(folds),
        os.path.join(args.output_dir, "split_protocol_table"),
        "Contribution 4 - split-protocol sensitivity (NEGATIVE CONTROL)",
        NEGATIVE_CONTROL_CAPTION,
    )

    print("done")


if __name__ == "__main__":
    main()
