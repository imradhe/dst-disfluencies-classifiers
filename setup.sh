#!/usr/bin/env bash
# ---------------------------------------------------------------------
# setup.sh - build the Disfluency Classifier v2 environment from scratch
# ---------------------------------------------------------------------
# Creates ./.venv (Python 3.11), installs everything, verifies imports,
# writes requirements-lock.txt. Idempotent; --recreate wipes .venv first.
#
#   ./setup.sh                      # CPU / Apple-MPS torch (default index)
#   ./setup.sh --cuda cu124         # Linux GPU box: CUDA 12.4 torch wheels
#   ./setup.sh --cuda cu121         # CUDA 12.1  (also cu126, cu128, ...)
#   ./setup.sh --recreate           # delete .venv and rebuild
#   ./setup.sh --no-optional        # skip gammatone + imbalanced-learn
#
# After it finishes:
#   . .venv/bin/activate
#   python run_mfcc_experiments.py --dry-run
# ---------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

CUDA=""
RECREATE=0
OPTIONAL=1
while [ $# -gt 0 ]; do
  case "$1" in
    --cuda)       CUDA="${2:-}"; shift 2;;
    --cuda=*)     CUDA="${1#*=}"; shift;;
    --recreate)   RECREATE=1; shift;;
    --no-optional) OPTIONAL=0; shift;;
    -h|--help)    sed -n '2,20p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

# --- pick a Python 3.11 interpreter --------------------------------
PY=""
for c in python3.11 python3.12 python3 python; do
  command -v "$c" >/dev/null 2>&1 || continue
  v="$("$c" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo x)"
  if [ "$v" = "3.11" ]; then PY="$c"; break; fi
  [ -z "$PY" ] && [ "$v" != "x" ] && PY="$c"
done
[ -z "$PY" ] && { echo "ERROR: no python3 found on PATH"; exit 1; }
PYVER="$("$PY" -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
echo ">> interpreter: $PY ($("$PY" --version 2>&1))"
[ "$PYVER" = "3.11" ] || echo "   WARNING: 3.11 is the tested version; continuing on $PYVER"

# --- venv --------------------------------------------------------
if [ "$RECREATE" = 1 ] && [ -d .venv ]; then echo ">> removing existing .venv"; rm -rf .venv; fi
[ -d .venv ] || { echo ">> creating .venv"; "$PY" -m venv .venv; }
PIP() { .venv/bin/python -m pip "$@"; }
PIP install -q -U pip wheel setuptools
echo ">> $(.venv/bin/python -m pip -V)"

# --- torch first (so the resolve keeps this exact build) --------
if [ -n "$CUDA" ]; then
  echo ">> torch/torchaudio for CUDA $CUDA"
  PIP install --timeout 180 "torch==2.13.0" "torchaudio==2.11.0" \
      --index-url "https://download.pytorch.org/${CUDA}"
else
  echo ">> torch/torchaudio (default index: CPU / Apple-MPS)"
  PIP install --timeout 180 "torch==2.13.0" "torchaudio==2.11.0"
fi

# --- core ------------------------------------------------------
echo ">> core requirements"
PIP install --timeout 180 -r requirements-core.txt

# --- optional ------------------------------------------------
if [ "$OPTIONAL" = 1 ]; then
  echo ">> optional requirements (gammatone [git], imbalanced-learn)"
  if ! PIP install --timeout 180 -r requirements-optional.txt; then
    echo "   !! optional install failed (often the gammatone git clone)."
    echo "      prosody32/96 features and paper_repro SMOTE will be unavailable."
    echo "      The MFCC x {rf,dnn,bilstm} grid does NOT need them."
    echo "      Retry later:  .venv/bin/python -m pip install -r requirements-optional.txt"
  fi
fi

# --- verify --------------------------------------------------
echo ">> verifying imports"
.venv/bin/python - <<'PYEOF'
import importlib, sys
core = ["numpy","scipy","sklearn","pandas","matplotlib","joblib","tqdm",
        "torch","torchaudio","soundfile","librosa","parselmouth"]
opt  = ["gammatone.gtgram","imblearn"]
bad = 0
for m in core:
    try:
        mod = importlib.import_module(m)
        print(f"  {m:14} OK   {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  {m:14} FAIL {e}"); bad += 1
for m in opt:
    try:
        importlib.import_module(m); print(f"  {m:20} OK   (optional)")
    except Exception as e:
        print(f"  {m:20} --   (optional, not installed)")
try:
    import classifier.runner, classifier.models.bilstm  # noqa: F401
    print("  classifier.*   OK")
except Exception as e:
    print(f"  classifier.*   FAIL {e}"); bad += 1
if bad:
    print(f"\n{bad} required import(s) failed"); sys.exit(1)
import torch
print("\n  torch.cuda.is_available():", torch.cuda.is_available(),
      "| torch.backends.mps.is_available():", torch.backends.mps.is_available())
PYEOF

echo ">> pipeline dry-run:"
if .venv/bin/python run_mfcc_experiments.py --dry-run >/dev/null 2>&1; then
  echo "   run_mfcc_experiments.py --dry-run: OK"
else
  echo "   run_mfcc_experiments.py --dry-run: FAILED or IED/ dataset not in place yet"
fi

echo ">> writing requirements-lock.txt"
{ echo "# pip freeze of the .venv built by setup.sh - $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# Platform: $(uname -srm)   Python: $(.venv/bin/python -V 2>&1)   --cuda: ${CUDA:-none}"
  .venv/bin/python -m pip freeze
} > requirements-lock.txt

echo
echo "DONE.  . .venv/bin/activate  &&  python run_mfcc_experiments.py"
