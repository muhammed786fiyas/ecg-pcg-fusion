"""Render the paper's figures: ROC curves, training curves, wavelet sensitivity,
scalogram examples and a pipeline diagram.

Reads the per-fold test_predictions.csv and training_history.csv that the
training scripts wrote, plus the results tables, so it does not reload a model.

The jet-colormap scalogram renders live here and nowhere else: the model input
is a uint8 memmap, and colour is only ever for human eyes.
"""

import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve

UINT8_MAX = 255.0
N_SCALOGRAM_EXAMPLES = 6
NEGATIVE_CONTROL_LABELS = ["dual_cnn_leaky_val", "dual_cnn_fully_leaky"]


def fold_prediction_files(report_root, family, config):
    base = os.path.join(report_root, family, config)
    if not os.path.isdir(base):
        return []
    found = []
    for name in sorted(os.listdir(base)):
        path = os.path.join(base, name, "test_predictions.csv")
        if os.path.exists(path):
            found.append((name, path))
    return found


def plot_roc_curves(report_root, config, families, out_path):
    """One mean ROC per family, with the per-fold spread shaded."""
    figure, axis = plt.subplots(figsize=(7, 6))
    grid = np.linspace(0, 1, 200)
    plotted = 0

    for family in families:
        files = fold_prediction_files(report_root, family, config)
        if len(files) == 0:
            continue
        curves = []
        aucs = []
        for _, path in files:
            frame = pd.read_csv(path)
            if frame["label"].nunique() < 2:
                continue
            false_positive, true_positive, _ = roc_curve(frame["label"], frame["prob"])
            curves.append(np.interp(grid, false_positive, true_positive))
            aucs.append(auc(false_positive, true_positive))
        if len(curves) == 0:
            continue

        mean_curve = np.mean(curves, axis=0)
        mean_curve[0] = 0.0
        label = f"{family} (AUC {round(float(np.mean(aucs)), 3)}"
        label = label + (f" ± {round(float(np.std(aucs)), 3)})" if len(aucs) > 1 else ")")
        axis.plot(grid, mean_curve, label=label, linewidth=2)
        if len(curves) > 1:
            spread = np.std(curves, axis=0)
            axis.fill_between(grid, np.clip(mean_curve - spread, 0, 1), np.clip(mean_curve + spread, 0, 1), alpha=0.15)
        plotted = plotted + 1

    if plotted == 0:
        plt.close(figure)
        print("  no predictions found, skipping ROC figure")
        return False

    axis.plot([0, 1], [0, 1], "k--", linewidth=1, label="chance")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("Segment-level ROC, record-level 5-fold CV (mean ± std)")
    axis.legend(loc="lower right", fontsize=8)
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    print(f"  wrote {out_path} ({plotted} families)")
    return True


def plot_training_curves(report_root, config, family, out_path):
    files = []
    base = os.path.join(report_root, family, config)
    if not os.path.isdir(base):
        print(f"  no history for {family}, skipping")
        return False
    for name in sorted(os.listdir(base)):
        path = os.path.join(base, name, "training_history.csv")
        if os.path.exists(path):
            files.append((name, path))
    if len(files) == 0:
        print(f"  no history for {family}, skipping")
        return False

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for fold_tag, path in files:
        history = pd.read_csv(path)
        axes[0].plot(history["epoch"], history["train_loss"], label=fold_tag + " train", alpha=0.8)
        axes[0].plot(history["epoch"], history["val_loss"], linestyle="--", label=fold_tag + " val", alpha=0.8)
        axes[1].plot(history["epoch"], history["val_auc"], label=fold_tag, alpha=0.9)

    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("BCE loss")
    axes[0].set_title(f"{family}: loss")
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("inner-validation AUC")
    axes[1].set_title(f"{family}: validation AUC (early stopping criterion)")
    axes[1].legend(fontsize=7)

    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    print(f"  wrote {out_path}")
    return True


def plot_wavelet_sensitivity(table_path, out_path):
    if not os.path.exists(table_path):
        print("  no wavelet table yet, skipping")
        return False
    table = pd.read_csv(table_path)
    if len(table) == 0:
        return False

    def parse_mean(value):
        text = str(value)
        if text in ("--", "nan"):
            return np.nan
        return float(text.split("±")[0].strip())

    def parse_std(value):
        text = str(value)
        if "±" not in text:
            return 0.0
        return float(text.split("±")[1].strip())

    means = [parse_mean(v) for v in table["Seg AUC"]]
    stds = [parse_std(v) for v in table["Seg AUC"]]
    labels = [f"{row['ECG wavelet']}\n/ {row['PCG wavelet']}" for _, row in table.iterrows()]

    figure, axis = plt.subplots(figsize=(max(7, len(labels) * 1.3), 5))
    positions = np.arange(len(labels))
    axis.bar(positions, means, yerr=stds, capsize=4, color="#4C78A8")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, fontsize=8)
    axis.set_ylabel("Segment AUC (mean ± std over folds)")
    axis.set_title("Contribution 1: CWT mother wavelet sensitivity (dual_cnn, scales held fixed)")
    finite = [v for v in means if not np.isnan(v)]
    if len(finite):
        axis.set_ylim(max(0.0, min(finite) - 0.1), min(1.0, max(finite) + 0.05))
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    print(f"  wrote {out_path}")
    return True


def render_scalogram_examples(scalogram_dir, manifest_path, out_dir, n_examples):
    """The only place jet-colormap PNGs are produced. For human eyes only."""
    if not os.path.isdir(scalogram_dir) or not os.path.exists(manifest_path):
        print("  scalograms or manifest missing, skipping examples")
        return 0
    os.makedirs(out_dir, exist_ok=True)

    manifest = pd.read_csv(manifest_path)
    ecg_memmap = np.load(os.path.join(scalogram_dir, "ecg.uint8.npy"), mmap_mode="r")
    pcg_memmap = np.load(os.path.join(scalogram_dir, "pcg.uint8.npy"), mmap_mode="r")

    normal = manifest[manifest["label"] == 0].head(n_examples // 2)
    abnormal = manifest[manifest["label"] == 1].head(n_examples - len(normal))
    chosen = pd.concat([normal, abnormal])

    written = 0
    for row in chosen.itertuples(index=False):
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(np.asarray(ecg_memmap[int(row.row_index)]) / UINT8_MAX, cmap="jet", aspect="auto", origin="lower")
        axes[0].set_title("ECG scalogram (cmor1.5-1.0, 4-100 Hz)")
        axes[1].imshow(np.asarray(pcg_memmap[int(row.row_index)]) / UINT8_MAX, cmap="jet", aspect="auto", origin="lower")
        axes[1].set_title("PCG scalogram (morl, 12.5-232 Hz)")
        for axis in axes:
            axis.set_xlabel("time (3 s window)")
            axis.set_ylabel("scale")
        label_name = "abnormal" if int(row.label) == 1 else "normal"
        figure.suptitle(f"{row.segment_variant_id} - record {row.record_id} - {label_name}")
        figure.tight_layout()
        figure.savefig(os.path.join(out_dir, f"{label_name}_{row.segment_variant_id}.png"), dpi=130)
        plt.close(figure)
        written = written + 1

    print(f"  wrote {written} scalogram example PNGs to {out_dir}")
    return written


def write_pipeline_diagram(out_path):
    """Mermaid, so it renders in the repo and can be redrawn for the paper."""
    diagram = """# Pipeline and architecture

## Data pipeline

```mermaid
flowchart TD
    A["raw WFDB<br/>409 records"] --> B["01_convert<br/>405 dual-modality"]
    B --> C["02_record_qc<br/>405 pass"]
    C --> D["03_segment<br/>R-peak-centred 3 s<br/>3752 segments"]
    D --> E["04_segment_qc<br/>3752 pass, 194 repaired"]
    C --> F["05_assign_folds<br/>RECORD-LEVEL, seeded"]
    E --> G["06_augment<br/>4x = 15008 rows"]
    G --> H["01_scalogram<br/>CWT -> uint8 memmap"]
    F --> I["build_manifests"]
    H --> I
    I --> J["training<br/>manifests only"]
```

Fold assignment sees only `(record_id, label)` pairs and a seed. Segmentation
runs before it, which is safe because segmentation is per-record deterministic
with no RNG.

## Model

```mermaid
flowchart LR
    E1["ECG scalogram<br/>1x224x224"] --> E2["CNNBranch<br/>4 conv blocks<br/>14x14x256"]
    P1["PCG scalogram<br/>1x224x224"] --> P2["CNNBranch<br/>4 conv blocks<br/>14x14x256"]
    E2 --> X["196 tokens x 256"]
    P2 --> Y["196 tokens x 256"]
    X --> Z["bidirectional<br/>cross-attention<br/>4 heads"]
    Y --> Z
    Z --> M["mean-pool + concat<br/>512-d"]
    M --> C["Linear 512-128<br/>ReLU, Dropout 0.5<br/>Linear 128-1"]
```
"""
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(diagram)
    print(f"  wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="render the paper's figures")
    parser.add_argument("--report-root", default="reports")
    parser.add_argument("--scalogram-root", default="data/processed/scalograms")
    parser.add_argument("--scalogram-config", default="default")
    parser.add_argument("--manifest-root", default="data/processed/manifests")
    parser.add_argument("--output-dir", default="reports/figures")
    args = parser.parse_args()

    print("=== 04_build_figures ===")
    os.makedirs(args.output_dir, exist_ok=True)

    families = [name for name, _, _, _ in [
        ("ecg_only", 0, 0, 0), ("pcg_only", 0, 0, 0), ("dual_cnn", 0, 0, 0),
        ("warm_start_fusion", 0, 0, 0), ("cbam_fusion", 0, 0, 0),
        ("cross_attn_fusion", 0, 0, 0), ("cross_attn_resnet18", 0, 0, 0),
        ("resnet18_ecg_only", 0, 0, 0), ("resnet18_pcg_only", 0, 0, 0),
    ]]

    print("ROC curves:")
    plot_roc_curves(args.report_root, args.scalogram_config, families,
                    os.path.join(args.output_dir, "roc_curves.png"))

    print("negative-control ROC (labelled as such):")
    plot_roc_curves(args.report_root, args.scalogram_config,
                    ["dual_cnn", *NEGATIVE_CONTROL_LABELS],
                    os.path.join(args.output_dir, "roc_curves_NEGATIVE_CONTROL.png"))

    print("training curves:")
    for family in ["cross_attn_fusion", "dual_cnn"]:
        plot_training_curves(args.report_root, args.scalogram_config, family,
                             os.path.join(args.output_dir, f"training_curves_{family}.png"))

    print("wavelet sensitivity:")
    plot_wavelet_sensitivity(os.path.join(args.output_dir, "wavelet_table.csv"),
                             os.path.join(args.output_dir, "wavelet_sensitivity.png"))

    print("scalogram examples:")
    render_scalogram_examples(
        os.path.join(args.scalogram_root, args.scalogram_config),
        os.path.join(args.manifest_root, "cv_fold0", "test.csv"),
        os.path.join(args.output_dir, "scalogram_examples"),
        N_SCALOGRAM_EXAMPLES,
    )

    print("pipeline diagram:")
    write_pipeline_diagram(os.path.join(args.output_dir, "pipeline_architecture.md"))
    print("done")


if __name__ == "__main__":
    main()
