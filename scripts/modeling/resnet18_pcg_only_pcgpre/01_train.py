"""Train resnet18_pcg_only_pcgpre.

resnet18_pcg_only, with its PCG branch initialised from the ResNet-18 PCG branch
pretrained on PhysioNet 2016 subsets B-F (scripts/modeling/pcg_pretrain/,
added 2026-09-12) instead of from ImageNet. Everything else - data, folds,
schedule, seed, head - is byte-identical to resnet18_pcg_only, so the pair isolates the
effect of PCG pretraining. B-F share no patients with Training-A.

The ResNet-18 unimodal baseline: one pretrained ResNet-18 branch on the PCG
scalogram, otherwise identical to cross_attn_resnet18. Added 2026-09-11 to test
directly whether fusion helps at the ResNet level - the custom-CNN baselines
cannot answer that, because their backbone differs from the best model's.

Self-contained by design. The Dataset class and the model definitions are
duplicated across every training script rather than imported from a sibling,
because that is what lets a Kaggle kernel be THIS FILE with its path constants
pointed at /kaggle/input/<slug>/ and /kaggle/working/. See
docs/logs/tasks/3-modeling.md.

Paths come from environment variables with local defaults, so one file runs in
both places without forking.

Reads manifests and nothing else. Never globs a directory.
"""

import argparse
import json
import os

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import yaml
from dotenv import load_dotenv
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

MODEL_FAMILY = "resnet18_pcg_only_pcgpre"
UINT8_MAX = 255.0
IMAGE_SIZE = 224

SCALOGRAM_ROOT = os.environ.get("SCALOGRAM_ROOT", "data/processed/scalograms")
MANIFEST_ROOT = os.environ.get("MANIFEST_ROOT", "data/processed/manifests")
MODEL_ROOT = os.environ.get("MODEL_ROOT", "models")
REPORT_ROOT = os.environ.get("REPORT_ROOT", "reports")

ECG_MEMMAP_NAME = "ecg.uint8.npy"
PCG_MEMMAP_NAME = "pcg.uint8.npy"


def resolve_wavelets(params, config_name):
    """Which wavelet pair a named scalogram config was built from.

    The ablation configs live in params.yaml; anything else is the default pair.
    Logging the config's own wavelets rather than the defaults is what makes the
    wavelet-sensitivity table readable straight off the MLflow compare view.
    """
    feature_params = params["features"]["scalogram"]
    for config in params["features"]["wavelet_ablation"]["configs"]:
        if config["name"] == config_name:
            return config["ecg_wavelet"], config["pcg_wavelet"]
    return feature_params["ecg_wavelet"], feature_params["pcg_wavelet"]


def load_params(params_path):
    with open(params_path) as handle:
        params = yaml.safe_load(handle)
    return params


class ScalogramDataset(Dataset):
    """Rows named by a manifest, read from the uint8 memmaps.

    The manifest is the only source of row membership. Opening the memmap lazily
    per worker avoids pickling a large mapped array into every DataLoader worker.
    """

    def __init__(self, manifest_path, scalogram_dir):
        self.manifest = pd.read_csv(manifest_path)
        self.ecg_path = os.path.join(scalogram_dir, ECG_MEMMAP_NAME)
        self.pcg_path = os.path.join(scalogram_dir, PCG_MEMMAP_NAME)
        self.rows = self.manifest["row_index"].to_numpy()
        self.labels = self.manifest["label"].to_numpy().astype(np.float32)
        self.ecg_memmap = None
        self.pcg_memmap = None

    def __len__(self):
        return len(self.manifest)

    def _ensure_open(self):
        if self.ecg_memmap is None:
            self.ecg_memmap = np.load(self.ecg_path, mmap_mode="r")
            self.pcg_memmap = np.load(self.pcg_path, mmap_mode="r")

    def __getitem__(self, position):
        self._ensure_open()
        row = int(self.rows[position])
        ecg = np.asarray(self.ecg_memmap[row], dtype=np.float32) / UINT8_MAX
        pcg = np.asarray(self.pcg_memmap[row], dtype=np.float32) / UINT8_MAX
        ecg_tensor = torch.from_numpy(ecg).unsqueeze(0)
        pcg_tensor = torch.from_numpy(pcg).unsqueeze(0)
        return ecg_tensor, pcg_tensor, torch.tensor(self.labels[position])


class ResNetBranch(nn.Module):
    """Pretrained ResNet-18 with the final fc removed - the SAME branch that
    cross_attn_resnet18 uses for each modality.

    The scalogram is 1-channel; it is replicated to 3 at forward time, which is
    the standard way to feed a grayscale field to RGB-pretrained weights. The
    512-channel map is projected to 256 exactly as in the fusion model.
    """

    def __init__(self, proj_dim):
        super().__init__()
        backbone = torchvision.models.resnet18(weights="DEFAULT")
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )
        self.project = nn.Conv2d(512, proj_dim, 1)

    def feature_maps(self, x):
        return self.project(self.stem(x.repeat(1, 3, 1, 1)))

    def forward(self, x):
        return self.feature_maps(x).mean(dim=(2, 3))


class ResNet18PcgOnly(nn.Module):
    """One pretrained ResNet-18 branch on the PCG scalogram.

    The direct test of whether fusion helps at the ResNet level. It matches
    cross_attn_resnet18 in backbone, projection, classifier head, learning rate,
    schedule and seed; the only things removed are the ECG branch and
    the cross-attention. Takes both tensors and ignores the ECG one so
    the Dataset and the epoch loop stay identical to the fusion scripts.
    """

    def __init__(self, dropout, proj_dim):
        super().__init__()
        self.pcg_branch = ResNetBranch(proj_dim)
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1)
        )

    def forward(self, ecg, pcg):
        return self.classifier(self.pcg_branch(pcg)).squeeze(1)


def build_model(params):
    family = params["modeling"]["resnet18_pcg_only_pcgpre"]
    return ResNet18PcgOnly(params["modeling"]["common"]["dropout"], family["proj_dim"])


def load_pcg_init(model, path):
    """Initialise the PCG branch from the PhysioNet 2016 B-F pretraining checkpoint.

    Hard-fails rather than falling back to ImageNet weights: a silent fallback
    would train the un-pretrained model and report it under the pretrained name.
    Only pcg_branch weights are taken; the head and any ECG branch start as in
    the un-pretrained twin.
    """
    if path == "" or not os.path.exists(path):
        raise SystemExit(f"QC FAIL: PCG pretraining checkpoint not found: '{path}'")
    state = torch.load(path, map_location="cpu", weights_only=False)
    prefix = "pcg_branch."
    branch_state = {}
    for key, value in state["model"].items():
        if key.startswith(prefix):
            branch_state[key[len(prefix):]] = value
    if len(branch_state) == 0:
        raise SystemExit(f"QC FAIL: no pcg_branch weights in {path}")
    model.pcg_branch.load_state_dict(branch_state)
    print(f"PCG branch initialised from {path} ({len(branch_state)} tensors, "
          f"pretraining val_auc={round(float(state.get('val_auc', float('nan'))), 4)})")


def compute_pos_weight(manifest_path):
    """Class weight fitted on the fold's TRAINING manifest only.

    Fitting this on anything wider - the full dataset, or train plus val - is a
    quiet leak: the code runs fine and the number is just optimistic.
    """
    manifest = pd.read_csv(manifest_path)
    n_positive = int((manifest["label"] == 1).sum())
    n_negative = int((manifest["label"] == 0).sum())
    if n_positive == 0:
        raise SystemExit("QC FAIL: training manifest has no positive examples")
    return float(n_negative) / float(n_positive)


def run_epoch(model, loader, device, criterion, optimizer, use_amp, scaler):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    all_probs = []
    all_labels = []

    for ecg, pcg, labels in loader:
        ecg = ecg.to(device, non_blocking=True)
        pcg = pcg.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            if use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    logits = model(ecg, pcg)
                    loss = criterion(logits, labels)
            else:
                logits = model(ecg, pcg)
                loss = criterion(logits, labels)

        if training:
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        total_loss = total_loss + float(loss.item()) * len(labels)
        all_probs.append(torch.sigmoid(logits.detach().float()).cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    return total_loss / len(labels), probs, labels


def safe_auc(labels, probs):
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probs))


def segment_metrics(labels, probs, threshold):
    predictions = (probs >= threshold).astype(int)
    true_positive = int(((predictions == 1) & (labels == 1)).sum())
    true_negative = int(((predictions == 0) & (labels == 0)).sum())
    false_positive = int(((predictions == 1) & (labels == 0)).sum())
    false_negative = int(((predictions == 0) & (labels == 1)).sum())

    sensitivity = true_positive / float(true_positive + false_negative) if (true_positive + false_negative) else 0.0
    specificity = true_negative / float(true_negative + false_positive) if (true_negative + false_positive) else 0.0

    return {
        "accuracy": float((predictions == labels).mean()),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "auc": safe_auc(labels, probs),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision_abnormal": float(precision_score(labels, predictions, pos_label=1, zero_division=0)),
        "recall_abnormal": float(recall_score(labels, predictions, pos_label=1, zero_division=0)),
        "precision_normal": float(precision_score(labels, predictions, pos_label=0, zero_division=0)),
        "recall_normal": float(recall_score(labels, predictions, pos_label=0, zero_division=0)),
    }


def patient_metrics(record_ids, labels, probs, threshold):
    """Mean-probability aggregation. The full three-strategy comparison lives in
    scripts/evaluation/01_patient_aggregation.py."""
    frame = pd.DataFrame({"record_id": record_ids, "label": labels, "prob": probs})
    grouped = frame.groupby("record_id").agg(label=("label", "first"), prob=("prob", "mean")).reset_index()
    return segment_metrics(grouped["label"].to_numpy(), grouped["prob"].to_numpy(), threshold)


def checkpoint_path(model_dir, tag):
    return os.path.join(model_dir, tag + "_last.pth")


def best_path(model_dir, tag):
    return os.path.join(model_dir, tag + "_best.pth")


def already_finished(run_key):
    """Resume-safety: skip a (config, protocol, fold) whose MLflow run finished."""
    experiment = mlflow.get_experiment_by_name(os.environ.get("MLFLOW_EXPERIMENT_NAME", "ecg-pcg-fusion"))
    if experiment is None:
        return False
    found = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.run_key = '{run_key}'",
    )
    if len(found) == 0:
        return False
    return bool((found["status"] == "FINISHED").any())


def train_one_fold(model, params, manifest_dir, scalogram_dir, model_dir, device, max_epochs, tag, batch_size, lr):
    common = params["modeling"]["common"]
    patience = common["early_stopping_patience"]
    threshold = params["evaluation"]["decision_threshold"]

    train_manifest = os.path.join(manifest_dir, "train.csv")
    val_manifest = os.path.join(manifest_dir, "val.csv")
    test_manifest = os.path.join(manifest_dir, "test.csv")

    n_workers = min(8, os.cpu_count())
    train_loader = DataLoader(
        ScalogramDataset(train_manifest, scalogram_dir),
        batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        ScalogramDataset(val_manifest, scalogram_dir),
        batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=(device.type == "cuda"),
    )
    test_dataset = ScalogramDataset(test_manifest, scalogram_dir)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=(device.type == "cuda"),
    )

    pos_weight = compute_pos_weight(train_manifest)
    print(f"pos_weight fitted on the training manifest only: {round(pos_weight, 4)}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler() if use_amp else None
    print(f"device={device} amp={use_amp}")

    start_epoch = 0
    best_auc = -1.0
    epochs_no_improve = 0
    last_path = checkpoint_path(model_dir, tag)
    if os.path.exists(last_path):
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        best_auc = state["best_auc"]
        epochs_no_improve = state["epochs_no_improve"]
        print(f"resumed from {last_path} at epoch {start_epoch} best_val_auc={round(best_auc, 4)}")

    history = []
    for epoch in range(start_epoch, max_epochs):
        train_loss, train_probs, train_labels = run_epoch(model, train_loader, device, criterion, optimizer, use_amp, scaler)
        val_loss, val_probs, val_labels = run_epoch(model, val_loader, device, criterion, None, use_amp, scaler)

        train_auc = safe_auc(train_labels, train_probs)
        val_auc = safe_auc(val_labels, val_probs)
        print(f"epoch {epoch + 1}/{max_epochs} train_loss={round(train_loss, 4)} train_auc={round(train_auc, 4)} val_loss={round(val_loss, 4)} val_auc={round(val_auc, 4)}")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "train_auc": train_auc, "val_auc": val_auc})
        mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss, "train_auc": train_auc, "val_auc": val_auc}, step=epoch)

        improved = val_auc > best_auc
        if improved:
            best_auc = val_auc
            epochs_no_improve = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_auc": val_auc}, best_path(model_dir, tag))
        else:
            epochs_no_improve = epochs_no_improve + 1

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_auc": best_auc,
                "epochs_no_improve": epochs_no_improve,
            },
            last_path,
        )

        if epochs_no_improve >= patience:
            print(f"early stopping at epoch {epoch + 1}, no val AUC improvement for {patience} epochs")
            break

    # Evaluate the checkpoint early stopping chose, not the last one.
    best_state = torch.load(best_path(model_dir, tag), map_location=device, weights_only=False)
    model.load_state_dict(best_state["model"])
    test_loss, test_probs, test_labels = run_epoch(model, test_loader, device, criterion, None, use_amp, scaler)

    seg = segment_metrics(test_labels, test_probs, threshold)
    pat = patient_metrics(test_dataset.manifest["record_id"].to_numpy(), test_labels, test_probs, threshold)

    predictions = test_dataset.manifest[["segment_variant_id", "record_id", "label"]].copy()
    predictions["prob"] = test_probs
    return seg, pat, predictions, pd.DataFrame(history), len(history), best_auc


def main():
    parser = argparse.ArgumentParser(description=f"train {MODEL_FAMILY}")
    parser.add_argument("--protocol", default="cv", choices=["dev", "cv"])
    parser.add_argument("--fold", type=int, default=-1, help="CV fold; -1 means all folds")
    parser.add_argument("--scalogram-config", default="default")
    parser.add_argument("--scalogram-root", default=SCALOGRAM_ROOT)
    parser.add_argument("--manifest-root", default=MANIFEST_ROOT)
    parser.add_argument("--model-root", default=MODEL_ROOT)
    parser.add_argument("--report-root", default=REPORT_ROOT)
    parser.add_argument("--max-epochs", type=int, default=0, help="0 means take it from params.yaml")
    parser.add_argument("--variant-tag", default="", help="suffix distinguishing a run variant, e.g. a split-protocol arm")
    parser.add_argument("--negative-control", action="store_true", help="tag this run as a NEGATIVE CONTROL, not a result")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument(
        "--pcg-init",
        default=os.environ.get("PCG_INIT", os.path.join(MODEL_ROOT, "pcg_pretrain", "pcg_pretrain", "pretrain_best.pth")),
        help="PCG pretraining checkpoint; required - there is no fallback",
    )
    args = parser.parse_args()

    load_dotenv()
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./models/mlflow_tracking"))
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "ecg-pcg-fusion")
    mlflow.set_experiment(experiment_name)

    params = load_params(args.params)
    seed = params["global_seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    common = params["modeling"]["common"]
    family_params = params["modeling"][MODEL_FAMILY]
    lr = family_params["lr"]
    batch_size = common["batch_size"]
    max_epochs = common["max_epochs_main"]
    smoke = params["smoke"]
    if smoke["enabled"]:
        max_epochs = smoke["epochs"]
    if args.max_epochs > 0:
        max_epochs = args.max_epochs

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        torch.set_num_threads(os.cpu_count())

    ecg_wavelet, pcg_wavelet = resolve_wavelets(params, args.scalogram_config)
    family_label = MODEL_FAMILY + args.variant_tag
    control_tag = "true" if args.negative_control else "false"
    scalogram_dir = os.path.join(args.scalogram_root, args.scalogram_config)
    print(f"=== train {MODEL_FAMILY} ===")
    print(f"protocol={args.protocol} scalogram_config={args.scalogram_config} device={device} max_epochs={max_epochs}")
    print(f"family_label={family_label} manifest_root={args.manifest_root} negative_control={control_tag}")
    if args.negative_control:
        print("NEGATIVE CONTROL: this run deliberately uses a leaky split. It is not a result.")

    if args.protocol == "dev":
        fold_tags = ["dev"]
    else:
        k = smoke["cv_folds"] if smoke["enabled"] else params["evaluation"]["cv_folds"]
        if args.fold >= 0:
            fold_tags = ["cv_fold" + str(args.fold)]
        else:
            fold_tags = ["cv_fold" + str(i) for i in range(k)]

    parent_name = f"{family_label}_{args.protocol}_{args.scalogram_config}"
    parent_key = parent_name + "_parent"

    fold_results = []
    with mlflow.start_run(run_name=parent_name):
        mlflow.set_tag("run_key", parent_key)
        mlflow.set_tag("model_family", MODEL_FAMILY)
        mlflow.set_tag("family_label", family_label)
        mlflow.set_tag("protocol", args.protocol)
        mlflow.set_tag("is_negative_control", control_tag)

        for fold_tag in fold_tags:
            run_key = f"{family_label}|{args.protocol}|{fold_tag}|{args.scalogram_config}"
            if already_finished(run_key):
                print(f"skipping {run_key}, an MLflow run for it already finished")
                continue

            manifest_dir = os.path.join(args.manifest_root, fold_tag)
            model_dir = os.path.join(args.model_root, family_label, args.scalogram_config)
            report_dir = os.path.join(args.report_root, family_label, args.scalogram_config, fold_tag)
            os.makedirs(model_dir, exist_ok=True)
            os.makedirs(report_dir, exist_ok=True)

            print(f"--- {run_key} ---")
            model = build_model(params).to(device)
            load_pcg_init(model, args.pcg_init)

            with mlflow.start_run(run_name=f"{parent_name}_{fold_tag}", nested=True):
                mlflow.set_tag("run_key", run_key)
                mlflow.set_tag("model_family", MODEL_FAMILY)
                mlflow.set_tag("family_label", family_label)
                mlflow.set_tag("protocol", args.protocol)
                mlflow.set_tag("is_negative_control", control_tag)
                mlflow.log_params(
                    {
                        "model_family": MODEL_FAMILY,
                        "family_label": family_label,
                        "protocol": args.protocol,
                        "fold": fold_tag,
                        "ecg_wavelet": ecg_wavelet,
                        "pcg_wavelet": pcg_wavelet,
                        "backbone": "resnet18_pretrained",
                        "pcg_init": "physionet2016_bf_pretrained",
                        "fusion": "none",
                        "modality": "pcg",
                        "lr": lr,
                        "batch_size": batch_size,
                        "seed": seed,
                        "scalogram_config": args.scalogram_config,
                    }
                )

                seg, pat, predictions, history, epochs_run, best_val_auc = train_one_fold(
                    model, params, manifest_dir, scalogram_dir, model_dir, device,
                    max_epochs, f"{args.scalogram_config}_{fold_tag}", batch_size, lr,
                )

                mlflow.log_param("epochs_run", epochs_run)
                mlflow.log_metric("best_val_auc", best_val_auc)
                mlflow.log_metrics({"seg_" + name: value for name, value in seg.items()})
                mlflow.log_metrics({"patient_" + name: value for name, value in pat.items()})

                predictions.to_csv(os.path.join(report_dir, "test_predictions.csv"), index=False)
                history.to_csv(os.path.join(report_dir, "training_history.csv"), index=False)
                with open(os.path.join(report_dir, "metrics.json"), "w") as handle:
                    json.dump({"segment": seg, "patient": pat, "epochs_run": epochs_run}, handle, indent=2)

                print(f"{fold_tag} seg_auc={round(seg['auc'], 4)} patient_auc={round(pat['auc'], 4)} epochs_run={epochs_run}")
                fold_results.append({"fold": fold_tag, "seg": seg, "patient": pat})

        if len(fold_results) > 1:
            for level in ["seg", "patient"]:
                for metric in fold_results[0][level].keys():
                    values = [result[level][metric] for result in fold_results]
                    mlflow.log_metric(f"{level}_{metric}_mean", float(np.nanmean(values)))
                    mlflow.log_metric(f"{level}_{metric}_std", float(np.nanstd(values)))
            aucs = [result["seg"]["auc"] for result in fold_results]
            print(f"CV summary over {len(fold_results)} folds: seg_auc mean={round(float(np.nanmean(aucs)), 4)} std={round(float(np.nanstd(aucs)), 4)}")

    print("done")


if __name__ == "__main__":
    main()
