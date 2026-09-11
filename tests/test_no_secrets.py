"""No tracked file may contain something shaped like an API credential.

The repo is public and commits are pushed as soon as they are made, so a secret
that reaches a commit is published within seconds and must be treated as leaked.
This nearly happened: a HuggingFace write token was pasted into README.md by a
right-click paste into an open editor tab while logging in. It was caught before
any commit, but only because someone looked.

This test scans every tracked file for credential-shaped strings. A local
pre-commit hook does the same on staged changes, which is the check that
actually stops a leak - this test is the versioned backstop that also runs in CI.
"""

import os
import re
import subprocess

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATTERNS = {
    "HuggingFace token": re.compile(r"hf_[A-Za-z0-9]{30,}"),
    "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),
    "Kaggle legacy API key": re.compile(r'"key"\s*:\s*"[0-9a-f]{32}"'),
}
MAX_BYTES = 5 * 1024 * 1024


def tracked_files():
    try:
        output = subprocess.run(
            ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [line for line in output.splitlines() if line.strip()]


def test_no_credentials_in_tracked_files():
    hits = []
    for relative in tracked_files():
        path = os.path.join(REPO_ROOT, relative)
        if not os.path.isfile(path) or os.path.getsize(path) > MAX_BYTES:
            continue
        with open(path, "rb") as handle:
            text = handle.read().decode("utf-8", errors="ignore")
        for name, pattern in PATTERNS.items():
            if pattern.search(text):
                hits.append(f"{relative}: looks like a {name}")
    assert len(hits) == 0, (
        "Credential-shaped strings in tracked files - remove them AND revoke the "
        "credential, since the repo is public:\n  " + "\n  ".join(hits)
    )
