"""Export a trained model to TorchScript for the inference image.

Defaults to cross_attn_resnet18, the study's best family, which the service in
scripts/serving/app.py defines. The inference image carries these artifacts and
nothing else - no training code, no dataset, no DVC. Tracing rather than
scripting because the model is a plain feed-forward graph with no
data-dependent control flow.

Everything in the output directory is copied into the image, so any artifact
left there from an earlier export is reported: remove it, or it ships too.
"""

import argparse
import importlib.util
import os

import torch
import yaml

IMAGE_SIZE = 224
DEFAULT_OUTPUT_DIR = os.path.join("models", "serving")


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
    parser.add_argument("--family", default="cross_attn_resnet18")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="", help="defaults to models/serving/<family>.pt")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    if args.output == "":
        args.output = os.path.join(DEFAULT_OUTPUT_DIR, args.family + ".pt")

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
    # cannot both hold. Shipping the state dict as well costs a second copy of
    # the weights and keeps all three endpoints working; the TorchScript is still
    # there for graph-only serving where /explain is not needed.
    eager_path = os.path.join(os.path.dirname(args.output), args.family + "_eager.pth")
    torch.save({"model": model.state_dict()}, eager_path)
    eager_mb = round(os.path.getsize(eager_path) / (1024.0 * 1024.0), 2)
    print(f"wrote {eager_path} ({eager_mb} MB) so /explain works in the container")

    written = [os.path.basename(args.output), os.path.basename(eager_path)]
    stale = [name for name in sorted(os.listdir(os.path.dirname(args.output))) if name not in written]
    if len(stale) > 0:
        print(f"WARNING: other files in {os.path.dirname(args.output)} will ship in the image too: {stale}")


if __name__ == "__main__":
    main()
