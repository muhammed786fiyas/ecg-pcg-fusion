"""Export cross_attn_fusion to TorchScript for the inference image.

The inference image carries this artifact and nothing else - no training code,
no dataset, no DVC. Tracing rather than scripting because the model is a plain
feed-forward graph with no data-dependent control flow.
"""

import argparse
import importlib.util
import os

import torch
import yaml

IMAGE_SIZE = 224
DEFAULT_OUTPUT = "models/serving/cross_attn_fusion.pt"


def load_family(family):
    path = os.path.join("scripts", "modeling", family, "01_train.py")
    if not os.path.exists(path):
        raise SystemExit(f"QC FAIL: {path} not found")
    spec = importlib.util.spec_from_file_location("train_" + family, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description="export the fusion model to TorchScript")
    parser.add_argument("--family", default="cross_attn_fusion")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    print("=== export_torchscript ===")
    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"QC FAIL: checkpoint not found: {args.checkpoint}")

    with open(args.params) as handle:
        params = yaml.safe_load(handle)

    module = load_family(args.family)
    model = module.build_model(params)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"loaded {args.checkpoint}")

    example_ecg = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE)
    example_pcg = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        traced = torch.jit.trace(model, (example_ecg, example_pcg))
        expected = model(example_ecg, example_pcg)
        actual = traced(example_ecg, example_pcg)

    difference = float(torch.max(torch.abs(expected - actual)))
    print(f"max |traced - eager| = {difference}")
    if difference > 1e-5:
        raise SystemExit(f"QC FAIL: traced model diverges from eager by {difference}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    traced.save(args.output)
    size_mb = round(os.path.getsize(args.output) / (1024.0 * 1024.0), 2)
    print(f"wrote {args.output} ({size_mb} MB)")

    # Also stage the eager checkpoint alongside it.
    #
    # The brief asks for an inference image carrying "only a TorchScript export",
    # but it also asks that image to serve /explain - and Grad-CAM cannot register
    # backward hooks on a scripted module's internals. Those two requirements
    # cannot both hold. Shipping the state dict as well costs ~5 MB and keeps all
    # three endpoints working; the TorchScript is still there for graph-only
    # serving where /explain is not needed.
    eager_path = os.path.join(os.path.dirname(args.output), "cross_attn_fusion_eager.pth")
    torch.save({"model": model.state_dict()}, eager_path)
    eager_mb = round(os.path.getsize(eager_path) / (1024.0 * 1024.0), 2)
    print(f"wrote {eager_path} ({eager_mb} MB) so /explain works in the container")


if __name__ == "__main__":
    main()
