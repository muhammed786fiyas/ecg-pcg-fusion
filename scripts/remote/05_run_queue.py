"""Work a queue of training jobs through Kaggle's 2-concurrent-GPU-session limit.

Kaggle refuses a third batch GPU push with "Maximum batch GPU session count of 2
reached", so the 60-odd (family, fold) jobs this project needs cannot simply be
fired off. This keeps two in flight, and as each finishes it fetches the output,
merges it into the local MLflow store, and starts the next.

It also maintains the quota ledger that section 9 of the brief requires: there is
no Kaggle quota API, so each kernel's wall-clock duration is recorded and totted
up, and the queue stops launching when the weekly budget is close to spent.

This is an addition to the brief's 01-04 remote scripts. It exists because the
concurrency cap turns "push, poll, fetch" from a one-liner into a scheduler, and
because the quota ledger has to come from somewhere.

Job file format, one per line, '#' for comments:
    family fold max_epochs [config] [variant_tag] [negative_control]
e.g.
    dual_cnn 0 50
    dual_cnn 0 25 w2_morl_morl
    dual_cnn 0 50 default _leaky_val true
"""

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time

MAX_CONCURRENT = 2
POLL_SECONDS = 45
TERMINAL_STATES = ["complete", "error", "cancel"]
AUTH_MARKERS = ["401", "unauthorized", "authentication", "invalid token", "expired", "403", "forbidden"]
LEDGER_NAME = "kaggle_quota_ledger.csv"

# The split-protocol arms read deliberately leaky manifests. Everything else
# reads the correct record-level ones.
# Which auxiliary dataset a job needs mounted alongside the main one.
# warm_start_fusion initialises from the unimodal checkpoints, and any
# non-default scalogram config lives in the ablation dataset.
CHECKPOINT_DATASET = "ecg-pcg-fusion-checkpoints"
ABLATION_DATASET = "ecg-pcg-fusion-ablation"

MANIFEST_SUBDIR_FOR_TAG = {
    "_leaky_val": "manifests_negative_control/arm_b_leaky_val",
    "_fully_leaky": "manifests_negative_control/arm_c_fully_leaky",
}

# Call the interpreter running this file, and the kaggle CLI installed beside it,
# rather than whatever "python"/"kaggle" happen to resolve to on PATH. The project
# lives in a conda env that is not on PATH by default.
PYTHON = sys.executable
KAGGLE = os.path.join(os.path.dirname(sys.executable), "Scripts", "kaggle.exe")
if not os.path.exists(KAGGLE):
    KAGGLE = "kaggle"


def kaggle(command_args):
    result = subprocess.run([KAGGLE, *command_args], capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


def looks_like_auth_failure(text):
    lowered = text.lower()
    return any(marker in lowered for marker in AUTH_MARKERS)


def read_jobs(path):
    jobs = []
    with open(path) as handle:
        for raw in handle:
            line = raw.strip()
            if line == "" or line.startswith("#"):
                continue
            parts = line.split()
            jobs.append(
                {
                    "family": parts[0],
                    "fold": int(parts[1]),
                    "max_epochs": int(parts[2]),
                    "config": parts[3] if len(parts) > 3 else "default",
                    "variant_tag": parts[4] if len(parts) > 4 and parts[4] != "-" else "",
                    "negative_control": len(parts) > 5 and parts[5].lower() == "true",
                }
            )
    return jobs


def job_slug(job):
    parts = [
        "ecgpcg",
        job["family"].replace("_", "-"),
        job["config"].replace("_", "-"),
        ("cv-fold" + str(job["fold"])),
    ]
    if job["variant_tag"]:
        parts.append(job["variant_tag"].strip("_").replace("_", "-"))
    return "-".join(parts).lower()


def extra_dataset_for(job):
    """The second dataset this job needs, if any."""
    if job["family"] == "warm_start_fusion":
        return CHECKPOINT_DATASET
    if job["config"] != "default":
        return ABLATION_DATASET
    return ""


def make_kernel(job, username, dataset_slug, output_dir):
    command = [
        PYTHON, "scripts/remote/02_make_kernel.py",
        "--family", job["family"],
        "--protocol", "cv",
        "--fold", str(job["fold"]),
        "--scalogram-config", job["config"],
        "--max-epochs", str(job["max_epochs"]),
        "--username", username,
        "--dataset-slug", dataset_slug,
        "--output-dir", output_dir,
    ]
    if job["variant_tag"]:
        command = [*command, "--variant-tag", job["variant_tag"]]
    if job["variant_tag"] in MANIFEST_SUBDIR_FOR_TAG:
        command = [*command, "--manifest-subdir", MANIFEST_SUBDIR_FOR_TAG[job["variant_tag"]]]
    extra = extra_dataset_for(job)
    if extra:
        command = [*command, "--extra-dataset", extra]
    if job["negative_control"]:
        command = [*command, "--negative-control"]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit("QC FAIL: 02_make_kernel failed: " + result.stdout + result.stderr)


def push(job_dir):
    return kaggle(["kernels", "push", "-p", job_dir])


def status_of(reference):
    code, text = kaggle(["kernels", "status", reference])
    if looks_like_auth_failure(text):
        raise SystemExit(
            "KAGGLE AUTH FAILURE. The OAuth token has most likely expired.\n"
            "Fix: run `kaggle auth login` again. Do NOT fall back to CPU training.\n" + text[:400]
        )
    lowered = text.lower()
    for state in TERMINAL_STATES:
        if state in lowered:
            return state
    if "running" in lowered:
        return "running"
    return "queued"


def collect(reference, job_dir, local_reports, local_models):
    """Fetch a kernel's output and merge it, and do NOT let a noisy fetch exit
    code skip the merge.

    `kaggle kernels output` has been observed returning a non-zero exit code
    while still delivering every file, which silently dropped a completed fold
    from the results: the checkpoints and MLflow run sat on disk, the run never
    reached the local store, and nothing failed loudly. The merge is the real
    test of whether the fetch worked - it hard-fails when there are no runs - so
    attempt it regardless and report what the merge says.
    """
    out_dir = os.path.join(job_dir, "output")
    os.makedirs(out_dir, exist_ok=True)
    code, text = kaggle(["kernels", "output", reference, "-p", out_dir])
    if code != 0:
        print("  note: kaggle output returned " + str(code) + ", merging anyway")
    merge = subprocess.run(
        [PYTHON, "scripts/remote/04_merge_results.py", "--kernel-output", out_dir,
         "--local-reports", local_reports, "--local-models", local_models],
        capture_output=True, text=True,
    )
    if merge.returncode != 0:
        print("  WARNING: merge failed: " + (merge.stdout + merge.stderr).strip()[-400:])
        return False
    for line in merge.stdout.strip().splitlines()[-3:]:
        print("  " + line)
    return True


def kernel_gpu_seconds(job_dir):
    """True in-kernel runtime, read from the returned Kaggle log.

    The wall clock between push and terminal status is NOT the quota figure: it
    includes however long the job sat in Kaggle's scheduling queue. Measured on
    one run, wall clock was 35.6 min against 7.9 min actually spent in the
    kernel - over-counting by 4.5x. Kaggle's 30 h/week budget is GPU session
    time, so charging queue wait against it would stop the run at roughly a
    fifth of the real allowance and would put a badly wrong number in the
    paper's compute accounting.

    Returns 0.0 if the log is unavailable, and the caller falls back to wall
    clock - conservative, which is the right direction for a budget.
    """
    logs = glob.glob(os.path.join(job_dir, "output", "*.log"))
    if len(logs) == 0:
        return 0.0
    try:
        with open(logs[0]) as handle:
            events = json.load(handle)
    except (ValueError, OSError):
        return 0.0
    if len(events) == 0:
        return 0.0
    return float(events[-1].get("time", 0.0))


def append_ledger(path, row):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["date", "kernel", "family", "config", "fold", "state", "hours", "wall_clock_hours", "weekly_total"]
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="run a queue of Kaggle training jobs")
    parser.add_argument("--jobs", required=True, help="job list file")
    parser.add_argument("--username", default=os.environ.get("KAGGLE_USERNAME", ""))
    parser.add_argument("--dataset-slug", default=os.environ.get("KAGGLE_DATASET_SLUG", "ecg-pcg-fusion-scalograms"))
    parser.add_argument("--kernel-dir", default=".kaggle_kernels")
    parser.add_argument("--local-reports", default="reports")
    parser.add_argument("--local-models", default="models")
    parser.add_argument("--ledger", default=os.path.join("docs", "logs", LEDGER_NAME))
    parser.add_argument("--budget-hours", type=float, default=25.0,
                        help="stop launching new jobs past this weekly total")
    args = parser.parse_args()

    if args.username == "":
        raise SystemExit("QC FAIL: no Kaggle username. Set KAGGLE_USERNAME in .env or pass --username")

    jobs = read_jobs(args.jobs)
    print("=== 05_run_queue ===")
    print(f"{len(jobs)} jobs queued, max {MAX_CONCURRENT} concurrent, budget {args.budget_hours} h")

    os.makedirs(os.path.dirname(args.ledger), exist_ok=True)
    pending = list(jobs)
    running = []
    weekly_total = 0.0
    finished = 0
    failed = []

    while len(pending) > 0 or len(running) > 0:
        while len(running) < MAX_CONCURRENT and len(pending) > 0:
            if weekly_total >= args.budget_hours:
                print(f"STOP: weekly total {round(weekly_total, 2)} h has reached the {args.budget_hours} h budget")
                print(f"{len(pending)} jobs left unlaunched - record them under 'next up' in PROJECT_CONTEXT.md")
                pending = []
                break

            job = pending.pop(0)
            slug = job_slug(job)
            job_dir = os.path.join(args.kernel_dir, slug)
            make_kernel(job, args.username, args.dataset_slug, args.kernel_dir)
            code, text = push(job_dir)
            if "successfully pushed" not in text:
                if "Maximum batch GPU session count" in text:
                    # Another session is still occupying a slot; retry this job later.
                    pending.insert(0, job)
                    break
                if looks_like_auth_failure(text):
                    raise SystemExit("KAGGLE AUTH FAILURE on push. Run `kaggle auth login`.\n" + text[:400])
                print(f"  push failed for {slug}: {text.strip()[:200]}")
                failed.append(slug)
                continue

            reference = args.username + "/" + slug
            running.append({"job": job, "slug": slug, "dir": job_dir, "ref": reference, "started": time.time()})
            print(f"launched {slug} ({len(pending)} still queued)")

        if len(running) == 0 and len(pending) == 0:
            break

        if len(running) == 0:
            # Every slot is held by a session this process did not launch - a
            # kernel pushed earlier, or by hand. Wait for one to free up rather
            # than exiting with jobs still queued.
            print(f"all GPU slots busy elsewhere, {len(pending)} jobs waiting")

        time.sleep(POLL_SECONDS)

        still_running = []
        for entry in running:
            state = status_of(entry["ref"])
            if state not in TERMINAL_STATES:
                still_running.append(entry)
                continue

            wall_hours = round((time.time() - entry["started"]) / 3600.0, 4)
            fetched = False
            if state == "complete":
                fetched = collect(entry["ref"], entry["dir"], args.local_reports, args.local_models)

            # Charge GPU seconds, not queue wait, against the weekly budget.
            gpu_hours = round(kernel_gpu_seconds(entry["dir"]) / 3600.0, 4)
            hours = gpu_hours if gpu_hours > 0 else wall_hours
            weekly_total = round(weekly_total + hours, 4)
            finished = finished + 1
            print(
                f"[{state}] {entry['slug']} gpu={round(hours * 60, 1)} min "
                f"(wall {round(wall_hours * 60, 1)} min, weekly total {round(weekly_total, 2)} h)"
            )

            if state == "complete":
                if not fetched:
                    failed.append(entry["slug"])
            else:
                failed.append(entry["slug"])
                print(f"  see the Kaggle log: kaggle kernels output {entry['ref']}")

            append_ledger(args.ledger, {
                "date": time.strftime("%Y-%m-%d %H:%M"),
                "kernel": entry["slug"],
                "family": entry["job"]["family"],
                "config": entry["job"]["config"],
                "fold": entry["job"]["fold"],
                "state": state,
                "hours": hours,
                "wall_clock_hours": wall_hours,
                "weekly_total": weekly_total,
            })
        running = still_running

    print("")
    print(f"queue done: {finished} kernels finished, {len(failed)} failed, {round(weekly_total, 2)} GPU-hours used")
    if len(failed) > 0:
        print("failed: " + str(failed))
    with open(os.path.join(os.path.dirname(args.ledger), "queue_summary.json"), "w") as handle:
        json.dump({"finished": finished, "failed": failed, "hours": weekly_total}, handle, indent=2)


if __name__ == "__main__":
    main()
