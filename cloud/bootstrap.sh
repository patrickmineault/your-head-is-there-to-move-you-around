#!/bin/bash
# Bootstrap a GPU worker WITHOUT baking application code: pull the analysis repo +
# the two model source repos at runtime, install deps, then run one work unit.
#
# Usable three ways:
#   (a) GCP Batch "script runnable"  -> no Docker, no image registry (primary path)
#   (b) GCE VM startup-script
#   (c) Docker ENTRYPOINT            -> deps + model repos baked; repo still pulled
#
# Each step is idempotent, so when deps / model repos are already present (Docker
# layer or warm VM) it is a fast no-op.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/patrickmineault/your-head-is-there-to-move-you-around}"
REPO_REF="${REPO_REF:-vjepa-midway-extension}"
WORK="${WORK:-/opt/motion-model}"
export VJEPA2_ROOT="${VJEPA2_ROOT:-$WORK/third_party/vjepa2}"
export MIDWAY_ROOT="${MIDWAY_ROOT:-$WORK/third_party/midway-network}"

mkdir -p "$WORK/third_party"
cd "$WORK"

# 1. application code -- always pulled, never baked.
if [ ! -d repo ]; then
    git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" repo
fi

# 2. third-party model sources (idempotent).
[ -d "$VJEPA2_ROOT/.git" ] || git clone --depth 1 \
    https://github.com/facebookresearch/vjepa2.git "$VJEPA2_ROOT"
[ -d "$MIDWAY_ROOT/.git" ] || git clone --depth 1 \
    https://github.com/agentic-learning-ai-lab/midway-network.git "$MIDWAY_ROOT"

# 3. locate a python that has pip (Deep Learning VM ships conda at /opt/conda but
# it is not on the Batch shell PATH; the script runnable runs as a non-login root).
PY=""
for cand in /opt/conda/bin/python3 "$(command -v python3 2>/dev/null)" /usr/bin/python3; do
    if [ -n "$cand" ] && [ -x "$cand" ] && "$cand" -m pip --version >/dev/null 2>&1; then
        PY="$cand"; break
    fi
done
if [ -z "$PY" ]; then
    echo "no python with pip found; trying ensurepip on python3"
    PY="$(command -v python3 || echo /usr/bin/python3)"
    "$PY" -m ensurepip --upgrade || true
fi
export PATH="$(dirname "$PY"):$PATH"
export PYTHON="$PY"
echo "using python: $PY ($("$PY" --version 2>&1))"

# 4. python deps (no-op if already satisfied).
"$PY" -m pip install -q -r repo/cloud/requirements-modern.txt

# 5. run the unit of work (reads MODE/DATASET/... or MANIFEST_GCS + BATCH_TASK_INDEX).
cd repo
exec bash cloud/run_worker.sh
