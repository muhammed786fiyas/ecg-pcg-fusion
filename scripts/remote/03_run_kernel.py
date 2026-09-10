"""Push a kernel, poll until it finishes, fetch its output.

There is no interactive GPU shell on Kaggle - the loop is push, poll, fetch.

Distinguishes authentication failures from quota and runtime failures. An
expired OAuth token is a one-minute fix, and quietly falling back to CPU
training instead of saying so would burn hours for no reason.
"""

import argparse
import json
import os
import subprocess
import time

AUTH_MARKERS = ["401", "unauthorized", "authentication", "invalid token", "expired", "403", "forbidden"]
DONE_STATES = ["complete", "error", "cancelled", "cancelAcknowledged"]


def run_kaggle(command_args, check=False):
    result = subprocess.run(["kaggle", *command_args], capture_output=True, text=True)
    combined = result.stdout + result.stderr
    if check and result.returncode != 0:
        classify_and_raise(combined, result.returncode)
    return result, combined


def looks_like_auth_failure(text):
    lowered = text.lower()
    return any(marker in lowered for marker in AUTH_MARKERS)


def classify_and_raise(text, code):
    if looks_like_auth_failure(text):
        raise SystemExit(
            "KAGGLE AUTH FAILURE. The OAuth token has most likely expired.\n"
            "Fix: run `kaggle auth login` again. Do NOT fall back to CPU training for this.\n"
            "Kaggle said: " + text.strip()[:500]
        )
    raise SystemExit(f"kaggle call failed with code {code}: " + text.strip()[:500])


def push(job_dir):
    print(f"pushing kernel from {job_dir}")
    result, text = run_kaggle(["kernels", "push", "-p", job_dir], check=True)
    print(text.strip())
    return result


def read_status(kernel_ref):
    result, text = run_kaggle(["kernels", "status", kernel_ref])
    if result.returncode != 0 and looks_like_auth_failure(text):
        classify_and_raise(text, result.returncode)
    lowered = text.lower()
    for state in DONE_STATES:
        if state.lower() in lowered:
            return state, text
    if "running" in lowered:
        return "running", text
    if "queued" in lowered:
        return "queued", text
    return "unknown", text


def poll(kernel_ref, interval_s, timeout_s):
    """Poll on a sane interval. Kaggle sessions are capped at 9 hours."""
    started = time.time()
    last_state = ""
    while True:
        elapsed = time.time() - started
        if elapsed > timeout_s:
            raise SystemExit(f"timed out after {round(elapsed / 60.0, 1)} minutes waiting for {kernel_ref}")

        state, text = read_status(kernel_ref)
        if state != last_state:
            print(f"[{round(elapsed / 60.0, 1)} min] status: {state}")
            last_state = state

        if state in DONE_STATES:
            print(text.strip())
            return state, elapsed

        time.sleep(interval_s)


def fetch_output(kernel_ref, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"fetching output to {output_dir}")
    result, text = run_kaggle(["kernels", "output", kernel_ref, "-p", output_dir], check=True)
    print(text.strip())
    listing = os.listdir(output_dir)
    print(f"{len(listing)} entries returned: {listing[:20]}")
    return listing


def main():
    parser = argparse.ArgumentParser(description="push, poll and fetch one Kaggle kernel")
    parser.add_argument("--job-dir", required=True, help="folder holding run.py + kernel-metadata.json")
    parser.add_argument("--output-dir", default="", help="defaults to <job-dir>/output")
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument("--timeout-hours", type=float, default=9.5)
    parser.add_argument("--no-fetch", action="store_true")
    args = parser.parse_args()

    metadata_path = os.path.join(args.job_dir, "kernel-metadata.json")
    if not os.path.exists(metadata_path):
        raise SystemExit(f"QC FAIL: no kernel-metadata.json in {args.job_dir}")
    with open(metadata_path) as handle:
        metadata = json.load(handle)
    kernel_ref = metadata["id"]

    output_dir = args.output_dir if args.output_dir else os.path.join(args.job_dir, "output")

    print("=== 03_run_kernel ===")
    print(f"kernel={kernel_ref} gpu={metadata.get('enable_gpu')} machine={metadata.get('machine_shape')}")

    push(args.job_dir)
    state, elapsed = poll(kernel_ref, args.poll_interval, args.timeout_hours * 3600.0)

    hours = round(elapsed / 3600.0, 3)
    print(f"kernel finished in state '{state}' after {hours} GPU-hours of wall clock")
    print(f"QUOTA: log {hours} h against the weekly 30 h budget in docs/logs/tasks/5-mlops.md")

    if state != "complete":
        raise SystemExit(f"kernel ended in state '{state}', not 'complete'. Check the Kaggle log.")

    if not args.no_fetch:
        fetch_output(kernel_ref, output_dir)

    summary = {"kernel": kernel_ref, "state": state, "wall_clock_hours": hours}
    with open(os.path.join(args.job_dir, "run_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print("done")


if __name__ == "__main__":
    main()
