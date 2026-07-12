#!/usr/bin/env python
"""Preflight for the autonomous algorithm-v2 run (tools/run_autonomous_v2.sh).

Fails FAST, in your terminal, on anything that would silently sabotage a 12-hour
unattended run -- above all: git identity and git push credentials, without which
every 20-minute checkpoint would fail and a box crash would lose everything.

    python tools/preflight_v2.py          # exit 0 = safe to launch

Checks: git identity | git push auth (dry-run) | torch+CUDA+arch | bitsandbytes
(the N6 4-bit arm) | analysis deps | HF token + access to the GATED Mistral repo
(N5) and GPQA | free disk for ~60GB of weights | the 52-cell matrix data that
N1/N4 read.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GATED_MODEL = "mistralai/Mistral-Small-Instruct-2409"   # N5 (worst-cell token-budget arm)
OPEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"                 # N6 (both quantization arms)
WEIGHTS_GB_NEEDED = 80                                  # ~44GB Mistral + ~15GB Qwen + headroom

fails: list[str] = []
warns: list[str] = []


def ok(msg: str) -> None:
    print(f"  \033[32mPASS\033[0m  {msg}")


def bad(msg: str, fix: str) -> None:
    print(f"  \033[31mFAIL\033[0m  {msg}\n        fix: {fix}")
    fails.append(msg)


def warn(msg: str, note: str) -> None:
    print(f"  \033[33mWARN\033[0m  {msg}\n        {note}")
    warns.append(msg)


def sh(*args: str) -> tuple[int, str]:
    p = subprocess.run(args, cwd=REPO, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).strip()


print("\n=== 1. git — checkpointing depends on these ===")
rc_n, name = sh("git", "config", "user.name")
rc_e, email = sh("git", "config", "user.email")
if rc_n == 0 and name and rc_e == 0 and email:
    ok(f"git identity: {name} <{email}>")
else:
    bad("git identity not set — every checkpoint commit would fail",
        'git config user.name "Aditya Bhatt" && git config user.email "you@example.com"')

rc, out = sh("git", "push", "--dry-run", "origin", "HEAD:main")
low = out.lower()
if rc == 0:
    ok("git push authenticates and fast-forwards (dry-run) — checkpoints will reach the remote")
elif "fast-forward" in low or "rejected" in low or "behind" in low:
    # NOT an auth problem: credentials worked, the history is just stale.
    bad("git push rejected: this branch is BEHIND origin/main (credentials are fine)",
        "git pull --rebase origin main     # then re-run — no new token needed")
elif any(s in low for s in ("authentication", "could not read username", "403",
                            "permission denied", "password authentication")):
    bad("git push cannot AUTHENTICATE — checkpoints would never leave this box",
        "use a token remote (classic PAT needs the `repo` scope; fine-grained needs "
        "Contents: Read and write):\n"
        "             git remote set-url origin https://<GITHUB_PAT>@github.com/bhattadiCS/"
        "research-thesis-overthinking-boundary.git")
else:
    bad("git push failed for an unexpected reason — checkpoints may not reach the remote",
        f"investigate: git push --dry-run origin HEAD:main\n"
        f"             (git said: {out.splitlines()[-1] if out else 'no output'})")

print("\n=== 2. GPU stack ===")
try:
    import torch
    if not torch.cuda.is_available():
        bad("torch cannot see a GPU", "check the container's CUDA runtime / driver visibility")
    else:
        cap = torch.cuda.get_device_capability(0)
        archs = torch.cuda.get_arch_list()
        name_gpu = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        ok(f"torch {torch.__version__} | {name_gpu} | {total:.0f} GB | sm_{cap[0]}{cap[1]}")
        if f"sm_{cap[0]}{cap[1]}" not in archs:
            bad(f"this torch build has no kernels for sm_{cap[0]}{cap[1]} (built for: {', '.join(archs)})",
                "install a torch built for Blackwell, e.g. a cu130 wheel:\n"
                "             pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu130")
        else:
            ok(f"torch has native sm_{cap[0]}{cap[1]} kernels")
except ImportError:
    bad("torch is not installed (NOTE: requirements-colab.txt does NOT include torch — "
        "the old image shipped it preinstalled)",
        "install the Blackwell/CUDA-13 wheel (matches the sm_120 card + driver on this box):\n"
        "             pip install torch --index-url https://download.pytorch.org/whl/cu130\n"
        "             (cu128 wheels also carry sm_120 kernels if cu130 is unavailable)")

try:
    import bitsandbytes  # noqa: F401
    from transformers import BitsAndBytesConfig  # noqa: F401
    ok("bitsandbytes + BitsAndBytesConfig available (needed by the N6 4-bit arm)")
except Exception as exc:  # noqa: BLE001
    bad(f"4-bit stack unavailable ({type(exc).__name__}) — the N6 4-bit arm would abort",
        "pip install 'bitsandbytes>=0.49.0'")

missing = [m for m in ("pandas", "sklearn", "sympy", "transformers", "accelerate", "datasets")
           if not __import__("importlib").util.find_spec(m)]
if missing:
    bad(f"missing python deps: {', '.join(missing)}", "pip install -r requirements-colab.txt")
else:
    ok("pandas / sklearn / sympy / transformers / accelerate / datasets present")

print("\n=== 3. Hugging Face access ===")
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
tok_file = REPO / ".hf_token"
if not token and tok_file.exists():
    token = tok_file.read_text().strip()
    ok(".hf_token file found (autostart.sh exports it; export it yourself for a manual run)")
if not token:
    warn("no HF_TOKEN in env and no .hf_token file",
         "a cached `huggingface-cli login` may still work; the gated check below is authoritative")
try:
    from huggingface_hub import HfApi
    api = HfApi()
    for repo_id, why in ((GATED_MODEL, "N5 — GATED, needs an accepted license"),
                         (OPEN_MODEL, "N6 — both arms")):
        try:
            api.model_info(repo_id, token=token or True)
            ok(f"{repo_id} accessible ({why})")
        except Exception as exc:  # noqa: BLE001
            bad(f"cannot access {repo_id} ({type(exc).__name__}) — {why}",
                f"accept the license at https://huggingface.co/{repo_id} with the account that owns the "
                "token, then:\n             export HF_TOKEN=hf_xxx   (or: echo hf_xxx > .hf_token)")
except ImportError:
    warn("huggingface_hub not importable — could not verify model access", "pip install huggingface_hub")

print("\n=== 4. Disk + data ===")
free_gb = shutil.disk_usage(REPO).free / 1e9
cache = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE") or "~/.cache/huggingface"
if free_gb >= WEIGHTS_GB_NEEDED:
    ok(f"{free_gb:.0f} GB free (need ~{WEIGHTS_GB_NEEDED} GB for Mistral-22B + Qwen-7B weights)")
else:
    bad(f"only {free_gb:.0f} GB free; ~{WEIGHTS_GB_NEEDED} GB needed for weights",
        f"free space, or point the cache at the persistent volume: export HF_HOME=/workspace-persist/hf_cache "
        f"(currently: {cache})")
if "/workspace-persist" not in str(Path(os.path.expanduser(cache))):
    warn(f"HF cache is at {cache} (not on the persistent volume)",
         "weights re-download if the container is recreated: export HF_HOME=/workspace-persist/hf_cache")

cells = list((REPO / "research/outputs/experiment_matrix").glob("*/detector_comparison_by_run.csv"))
if len(cells) >= 52:
    ok(f"{len(cells)} analyzed cells present (N1/N4 read these)")
else:
    bad(f"only {len(cells)} analyzed cells found; N1/N4 need the full 52",
        "git pull  (the matrix CSVs are tracked in the repo)")

print()
if fails:
    print(f"\033[31mX PREFLIGHT FAILED — {len(fails)} blocking issue(s). Fix them, then re-run.\033[0m")
    sys.exit(1)
print(f"\033[32mOK PREFLIGHT PASSED\033[0m" + (f" ({len(warns)} warning(s))" if warns else "")
      + " — safe to launch the autonomous run.\n")
sys.exit(0)
