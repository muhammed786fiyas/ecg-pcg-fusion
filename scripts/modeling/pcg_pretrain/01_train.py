"""Pretrain the ResNet-18 PCG branch on PhysioNet 2016 subsets B-F.

Stage 2 of the PCG pretraining experiment (added 2026-09-12; params.yaml
pretrain). The model is resnet18_pcg_only's - the same ResNetBranch, 1x1
projection to 256 and head - trained on the B-F windows built by
scripts/pretrain/, with early stopping on the B-F validation records. The
checkpoint's pcg_branch weights then initialise the PCG branch of
resnet18_pcg_only_pcgpre and cross_attn_resnet18_pcgpre, which are fine-tuned
and evaluated on Training-A's record-level CV folds exactly like every other
family.

Nothing here reads Training-A. The validation AUC printed at the end is a
sanity check that pretraining learned something, not a result: it is measured
on the same hospitals the branch was trained on.

Device-agnostic, resume-safe and MLflow-logged like every training script, and
it takes the same command-line arguments, so the Kaggle entry script runs it
unmodified. --protocol and --fold are accepted and ignored: there is one
train/val partition, not a fold loop.
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
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

MODEL_FAMILY = "pcg_pretrain"
UINT8_MAX = 255.0
RUN_TAG = "pretrain"

SCALOGRAM_ROOT = os.environ.get("SCALOGRAM_ROOT", "data/processed/scalograms")
MANIFEST_ROOT = os.environ.get("MANIFEST_ROOT", "data/processed/manifests_pcg_pretrain")
MODEL_ROOT = os.environ.get("MODEL_ROOT", "models")
REPORT_ROOT = os.environ.get("REPORT_ROOT", "reports")

PCG_MEMMAP_NAME = "pcg.uint8.npy"


def load_params(params_path):
    with open(params_path) as handle:
        return yaml.safe_load(handle)


class PcgScalogramDataset(Dataset):
    """PCG-only rows named by a manifest, read from the uint8 memmap.

    Returns an empty ECG placeholder so the epoch loop and the model's
    (ecg, pcg) signature match every other training script.
    """

    def __init__(self, manifest_path, scalogram_dir):
        self.manifest = pd.read_csv(manifest_path)
        self.pcg_path = os.path.join(scalogram_dir, PCG_MEMMAP_NAME)
        self.rows = self.manifest["row_index"].to_numpy()
        self.labels = self.manifest["label"].to_numpy().astype(np.float32)
        self.pcg_memmap = None

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, position):
        if self.pcg_memmap is None:
            self.pcg_memmap = np.load(self.pcg_path, mmap_mode="r")
        pcg = np.asarray(self.pcg_memmap[int(self.rows[position])], dtype=np.float32) / UINT8_MAX
        return torch.zeros(1), torch.from_numpy(pcg).unsqueeze(0), torch.tensor(self.labels[position])


class ResNetBranch(nn.Module):
    """Pretrained ResNet-18 with the final fc removed - the branch every
    ResNet-18 family uses. 1-channel input replicated to 3; the 512-channel map
    projected to 256."""

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
    """resnet18_pcg_only's model, so its pcg_branch drops straight into the
    fine-tuned families. Ignores the ECG placeholder."""

    def __init__(self, dropout, proj_dim):
        super().__init__()
        self.pcg_branch = ResNetBranch(proj_dim)
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1)
        )

    def forward(self, ecg, pcg):
        return self.classifier(self.pcg_branch(pcg)).squeeze(1)


def build_model(params):
    family = params["modeling"]["pcg_pretrain"]
    return ResNet18PcgOnly(params["modeling"]["common"]["dropout"], family["proj_dim"])


def compute_pos_weight(manifest_path):
    """Class weight fitted on the pretraining TRAIN manifest only."""
    manifest = pd.read_csv(manifest_path)
    n_positive = int((manifest["label"] == 1).sum())
    n_negative = int((manifest["label"] == 0).sum())
    if n_positive == 0:
        raise SystemExit("QC FAIL: pretraining train manifest has no positive examples")
    return float(n_negative) / float(n_positive)


def run_epoch(model, loader, device, criterion, optimizer, use_amp, scaler):
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()
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


def record_auc(record_ids, labels, probs):
    frame = pd.DataFrame({"record_id": record_ids, "label": labels, "prob": probs})
    grouped = frame.groupby("record_id").agg(label=("label", "first"), prob=("prob", "mean"))
    return safe_auc(grouped["label"].to_numpy(), grouped["prob"].to_numpy())


def already_finished(run_key):
    experiment = mlflow.get_experiment_by_name(os.environ.get("MLFLOW_EXPERIMENT_NAME", "ecg-pcg-fusion"))
    if experiment is None:
        return False
    found = mlflow.search_runs(experiment_ids=[experiment.experiment_id], filter_string=f"tags.run_key = '{run_key}'")
    if len(found) == 0:
        return False
    return bool((found["status"] == "FINISHED").any())


def train(model, params, manifest_dir, scalogram_dir, model_dir, device, max_epochs, batch_size, lr):
    patience = params["modeling"]["common"]["early_stopping_patience"]
    train_manifest = os.path.join(manifest_dir, "train.csv")
    val_manifest = os.path.join(manifest_dir, "val.csv")
    for path in [train_manifest, val_manifest]:
        if not os.path.exists(path):
            raise SystemExit(f"QC FAIL: pretraining manifest not found: {path}")

    n_workers = min(8, os.cpu_count())
    pin = device.type == "cuda"
    train_loader = DataLoader(PcgScalogramDataset(train_manifest, scalogram_dir), batch_size=batch_size,
                              shuffle=True, num_workers=n_workers, pin_memory=pin)
    val_dataset = PcgScalogramDataset(val_manifest, scalogram_dir)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=pin)

    pos_weight = compute_pos_weight(train_manifest)
    print(f"pos_weight fitted on the pretraining train manifest only: {round(pos_weight, 4)}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    use_amp = device.type == "cuda"
    scaler = None
    if use_amp:
        scaler = torch.amp.GradScaler()
    print(f"device={device} amp={use_amp} train rows={len(train_loader.dataset)} val rows={len(val_dataset)}")

    last_path = os.path.join(model_dir, RUN_TAG + "_last.pth")
    best_path = os.path.join(model_dir, RUN_TAG + "_best.pth")
    start_epoch = 0
    best_auc = -1.0
    epochs_no_improve = 0
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
        print(f"epoch {epoch + 1}/{max_epochs} train_loss={round(train_loss, 4)} train_auc={round(train_auc, 4)} "
              f"val_loss={round(val_loss, 4)} val_auc={round(val_auc, 4)}")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
                        "train_auc": train_auc, "val_auc": val_auc})
        mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss, "train_auc": train_auc, "val_auc": val_auc}, step=epoch)

        if val_auc > best_auc:
            best_auc = val_auc
            epochs_no_improve = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_auc": val_auc}, best_path)
        else:
            epochs_no_improve = epochs_no_improve + 1
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                    "best_auc": best_auc, "epochs_no_improve": epochs_no_improve}, last_path)
        if epochs_no_improve >= patience:
            print(f"early stopping at epoch {epoch + 1}, no val AUC improvement for {patience} epochs")
            break

    best_state = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_state["model"])
    _, val_probs, val_labels = run_epoch(model, val_loader, device, criterion, None, use_amp, scaler)
    val_record_auc = record_auc(val_dataset.manifest["record_id"].to_numpy(), val_labels, val_probs)
    return pd.DataFrame(history), best_auc, val_record_auc, len(history)


def main():
    parser = argparse.ArgumentParser(description="pretrain the ResNet-18 PCG branch on PhysioNet 2016 B-F")
    parser.add_argument("--protocol", default="pretrain", help="accepted for the Kaggle entry script; ignored")
    parser.add_argument("--fold", type=int, default=-1, help="accepted for the Kaggle entry script; ignored")
    parser.add_argument("--scalogram-config", default="pcg_pretrain")
    parser.add_argument("--scalogram-root", default=SCALOGRAM_ROOT)
    parser.add_argument("--manifest-root", default=MANIFEST_ROOT, help="folder holding train.csv and val.csv")
    parser.add_argument("--model-root", default=MODEL_ROOT)
    parser.add_argument("--report-root", default=REPORT_ROOT)
    parser.add_argument("--max-epochs", type=int, default=0, help="0 means params.yaml pretrain.max_epochs")
    parser.add_argument("--variant-tag", default="", help="accepted for the Kaggle entry script; unused")
    parser.add_argument("--negative-control", action="store_true", help="accepted; pretraining is never one")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    load_dotenv()
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./models/mlflow_tracking"))
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "ecg-pcg-fusion"))

    params = load_params(args.params)
    seed = params["global_seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    lr = params["modeling"][MODEL_FAMILY]["lr"]
    batch_size = params["modeling"]["common"]["batch_size"]
    max_epochs = params["pretrain"]["max_epochs"]
    if args.max_epochs > 0:
        max_epochs = args.max_epochs

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        torch.set_num_threads(os.cpu_count())

    scalogram_dir = os.path.join(args.scalogram_root, args.scalogram_config)
    model_dir = os.path.join(args.model_root, MODEL_FAMILY, args.scalogram_config)
    report_dir = os.path.join(args.report_root, MODEL_FAMILY, args.scalogram_config, RUN_TAG)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    print(f"=== train {MODEL_FAMILY} ===")
    print(f"scalograms={scalogram_dir} manifests={args.manifest_root} device={device} max_epochs={max_epochs}")

    run_key = f"{MODEL_FAMILY}|{RUN_TAG}|{args.scalogram_config}"
    if already_finished(run_key):
        print(f"skipping {run_key}, an MLflow run for it already finished")
        return

    model = build_model(params).to(device)
    with mlflow.start_run(run_name=f"{MODEL_FAMILY}_{args.scalogram_config}"):
        mlflow.set_tag("run_key", run_key)
        mlflow.set_tag("model_family", MODEL_FAMILY)
        mlflow.set_tag("family_label", MODEL_FAMILY)
        mlflow.set_tag("protocol", RUN_TAG)
        mlflow.set_tag("is_negative_control", "false")
        mlflow.log_params({
            "model_family": MODEL_FAMILY, "family_label": MODEL_FAMILY, "protocol": RUN_TAG,
            "data": "physionet2016_training_b_to_f_pcg", "backbone": "resnet18_pretrained",
            "lr": lr, "batch_size": batch_size, "seed": seed, "scalogram_config": args.scalogram_config,
        })
        history, best_val_auc, val_record_auc, epochs_run = train(
            model, params, args.manifest_root, scalogram_dir, model_dir, device, max_epochs, batch_size, lr
        )
        mlflow.log_param("epochs_run", epochs_run)
        mlflow.log_metric("best_val_auc", best_val_auc)
        mlflow.log_metric("val_record_auc", val_record_auc)
        history.to_csv(os.path.join(report_dir, "training_history.csv"), index=False)
        with open(os.path.join(report_dir, "metrics.json"), "w") as handle:
            json.dump({"best_val_segment_auc": best_val_auc, "val_record_auc": val_record_auc,
                       "epochs_run": epochs_run}, handle, indent=2)
    print(f"pretraining done: best val segment AUC {round(best_val_auc, 4)}, "
          f"val record AUC {round(val_record_auc, 4)} (B-F hospitals - a sanity check, not a result)")


if __name__ == "__main__":
    main()
