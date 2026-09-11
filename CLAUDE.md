# CLAUDE.md — rules for this repo

ECG–PCG dual-branch CWT scalogram fusion with cross-modal attention, on
PhysioNet/CinC 2016 Training-A. Target venue: Biomedical Signal Processing and
Control (Elsevier). Owner: Muhammed Fiyas.

Read `PROJECT_CONTEXT.md` for status and history. This file holds the rules that
must not drift.

---

## Environment — every Python command needs the prefix

The project env is a dedicated conda env named `ecg_pcg` (Python 3.11.16).
Never `base`, never another env.

Each shell call is a fresh shell, so `conda activate` does **not** persist.
Prefix every Python-touching command:

```
conda run -n ecg_pcg python scripts/data/01_convert.py --output ...
conda run -n ecg_pcg dvc repro
conda run -n ecg_pcg python -m pytest tests/ -q
conda run -n ecg_pcg python -m pip install -r requirements.txt
```

Use `python -m pip`, never bare `pip` — the `pip.exe` launcher on this machine
fails with "Access is denied".

`cmd` entries in `dvc.yaml` stay as plain `python scripts/...`; they inherit the
environment from whatever invokes `dvc`.

`numpy<2` is pinned because `neurokit2==0.2.7` predates the NumPy 2 ABI break.

## Git

Commit after each numbered milestone in `KICKOFF_PROMPT.md` §15.

**Pushing is authorised.** The owner amended the original brief on 2026-09-10:
the kickoff prompt said "never push, I review and push myself"; the standing
instruction now is to push after committing. Push to `origin main` as work
completes.

Two cautions that came out of getting this wrong once:
- **Never commit a rebuild-on-demand archive.** `.gitignore` covers
  `.kaggle_staging*/`, `.kaggle_kernels/`, `payload.zip` and `*.zip`. A 430 MB
  payload once reached a commit because the pattern was `.kaggle_staging/` and
  the directory was `.kaggle_staging_ablation/`. GitHub hard-rejects any file
  over 100 MB.
- **A history rewrite diverges from the remote.** If `filter-branch` or similar
  is ever needed again, back the remote up to a local branch first
  (`git branch <name> origin/main`) before force-pushing, so the pre-rewrite
  commits survive.

- **Never commit a credential.** The repo is public and pushes are immediate,
  so a committed secret is leaked within seconds and must be revoked. A local
  `.git/hooks/pre-commit` blocks staged changes that add HuggingFace, GitHub or
  Kaggle-key-shaped strings - it is not versioned, so reinstall it on a fresh
  clone. `tests/test_no_secrets.py` is the versioned backstop. Never echo a
  token in tool output; redact it.

### Definition of done — check this before every `git commit`

> **A step is not complete until `PROJECT_CONTEXT.md`'s Status section and
> `docs/logs/daily/` are updated. Both go in the SAME commit as the step's work,
> before running `git commit`.**

Not "at the end", not "when there's a natural pause". In the same commit.

This is tied to the commit because committing is the one thing that reliably
happens at every step. §14 of the brief already asked for these to be kept
current, and it drifted anyway: `PROJECT_CONTEXT.md` sat seven steps stale,
still listing the data pipeline as "next up" long after the pipeline, the
scalograms, the tests and the Kaggle offload were all done. Documentation that
depends on remembering to do it does not survive a long session or a context
compaction.

What "updated" means:
- **`PROJECT_CONTEXT.md` Status** — what is actually done, what is in progress
  *right now*, what is next in order, and any open item a future session would
  otherwise rediscover the hard way. Real measured numbers, not estimates.
- **`docs/logs/daily/DAYn_DD-MM-YYYY.md`** — Work done / Key decisions /
  Discussed-not-decided / Blockers / Next. Add deviations from the brief and
  why, as they happen.
- If the step changed a decision or turned up a gotcha, the relevant
  `docs/logs/tasks/n-*.md` gets it too.

---

## Tooling gotcha — patching files from a shell heredoc

When a file is patched with a Python script inside a Bash heredoc, the shell
layer collapses a doubled backslash, so an intended backslash-n escape arrives
as a REAL newline and breaks the string literal it was meant to sit in. This has
broken four patches. Either use the Edit tool for anything containing escape
sequences, or build them in the script with `chr(92)` (backslash) and
`chr(10)` (newline). Always `ast.parse` or `ruff check` a patched file before
moving on.

---

## The split rules — the top correctness requirement

**The patient is the unit of splitting.** Every split — train, val, test, every
CV fold — is decided over *record IDs*, never over segments or augmented rows.

1. `scripts/data/05_assign_folds.py` may only ever see `(record_id, label)`
   pairs and a seed. Never a segment ID, a signal, or anything derived from one.
   This is structural, and `tests/test_split_inputs.py` asserts it.
2. Segmentation runs *before* fold assignment, on all records at once. That is
   safe: segmentation is per-record deterministic (`nk.ecg_peaks` + fixed
   non-overlapping windows, no RNG), so a record's segments are identical
   whichever fold it lands in. Running it first means folds are assigned over
   the records that actually survive QC.
3. **Two protocols.** `dev` = fold 0 only, one stratified record-level
   60/15/25 partition, for all exploration. `cv` = 5-fold stratified
   record-level CV, for every number in the paper. Report mean ± std across
   folds, never the best fold.
4. **Augmentation is training-portion only.** Inner-validation and test
   manifests carry `_orig` rows only. Enforced at manifest level.
5. **Anything fitted is fitted inside the fold's training portion** —
   `pos_weight`, any normalization statistic, any decision threshold moved off
   0.5. A global mean/std is a leak; per-record and per-image normalization are
   safe by construction.
6. **Manifests are the only thing a training script reads.** No training script
   is ever allowed to glob a directory.

`tests/test_leakage.py` is the guard. Run it before and after any refactor that
touches splitting.

---

## Coding conventions — every script, no exceptions

- Flat, beginner-friendly. Constants in CAPS at the top, one `main()` holding
  the flow, `argparse` for CLI inputs, `if __name__ == "__main__": main()` last.
- Helper functions for genuine units of work. List comprehensions for simple
  transforms. Classes only where there is real state (a `Dataset`, an
  `nn.Module`) — not for organization.
- `try/except` only with named exceptions, only where a specific error is
  genuinely expected. No blanket `except:`.
- `print()` progress as plain f-strings: what happened plus the key numbers. No
  alignment format specs, no emojis, no decorative separators. A plain
  `=== Section ===` header is fine.
- **Avoid**: walrus operator, `f"{x=}"`, ternary expressions, type hints,
  `requests.Session()` with retry/backoff.
- Must run on both Python 3.11 locally and Kaggle's ~3.11: no `match`
  statements, no `X | Y` union syntax.
- **No shared utils module.** Each script carries its own constants and helpers.
  The `Dataset` class and the model definitions are duplicated across the
  training scripts on purpose — that is what lets a training script be dropped
  into a Kaggle kernel unchanged. Recorded as a decision in
  `docs/logs/tasks/3-modeling.md`.
- Never hardcode absolute paths. Everything comes from `argparse` args or
  `params.yaml`, resolved relative to repo root.
- Script numbering: zero-padded sequential prefix per folder (`01_`, `02_`).
  One script = one pipeline stage = one `dvc.yaml` stage.
- `os.makedirs(os.path.dirname(args.output), exist_ok=True)` before any write.
- Every stage: **load → QC gate (hard-fail with `raise SystemExit`, log what
  failed) → compute/write**.

## Training-script rules

- **Device-agnostic.** Never call `torch.amp.autocast(device_type="cuda")` or
  `GradScaler()` unconditionally. AMP only when `torch.cuda.is_available()`,
  plain fp32 otherwise, so the identical file runs on local CPU and a Kaggle T4.
- **Resume-safe.** Re-running picks up from the last checkpoint and skips
  already-completed MLflow runs rather than restarting.
- Paths come from argparse defaults or env vars so one file runs both locally
  and against `/kaggle/input/<slug>/` + `/kaggle/working/`. Do **not** fork the
  training code into a local copy and a Kaggle copy.
- MLflow: `load_dotenv()` before any `set_experiment()`/`start_run()`, and
  `os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")` inside the script.

## Reporting

Report whatever the numbers actually are. The old pipeline's numbers (fusion
0.817 / ECG 0.795 / PCG 0.647) came from a patient-level-leaking validation
split and are **not** a target. A lower honest number is a better result than a
higher dishonest one. The §10.4 split-protocol rows are **negative controls** and
must be labelled as such in the CSV, the figure captions and the MLflow tags —
they must never be readable as a result of the method.
