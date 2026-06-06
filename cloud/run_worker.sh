#!/bin/bash
# Worker entrypoint for one unit of work (run on a GCE GPU VM / GCP Batch task /
# locally). A work unit is either:
#   MODE=extract  -> build the feature cache for (DATASET, FEATURES) [GPU]
#   MODE=fit      -> ridge-fit one neuron (DATASET, FEATURES, SUBSET) from cache
#
# The work unit is taken from explicit env vars, or, if MANIFEST is set, from the
# (0-based) line ${BATCH_TASK_INDEX} of a TSV manifest "MODE DATASET FEATURES SUBSET".
set -euo pipefail

BUCKET="${BUCKET:-gs://xcorr-dev-motion-model}"
DATA_ROOT="${DATA_ROOT:-/data/data_derived}"
CKPT_ROOT="${CKPT_ROOT:-/data/checkpoints}"
CACHE_ROOT="${CACHE_ROOT:-/cache}"
DEVICE="${DEVICE:-cuda}"
INPUT_ADAPT="${INPUT_ADAPT:-resize}"
EXP_NAME="${EXP_NAME:-vjepa_midway_fit}"

# GCP Batch path: the manifest lives in GCS; fetch it, then index by task id.
if [ -n "${MANIFEST_GCS:-}" ]; then
    gsutil -q cp "$MANIFEST_GCS" /tmp/manifest.tsv
    MANIFEST=/tmp/manifest.tsv
fi
if [ -n "${MANIFEST:-}" ]; then
    line=$(sed -n "$((${BATCH_TASK_INDEX:-0} + 1))p" "$MANIFEST")
    read -r MODE DATASET FEATURES SUBSET <<< "$line"
fi
: "${MODE:?}"; : "${DATASET:?}"; : "${FEATURES:?}"; SUBSET="${SUBSET:-0}"

# dataset train-name -> on-disk folder (matches loaders/get_dataset).
case "$DATASET" in
    pvc1-repeats)        FOLDER=crcns-pvc1 ;;
    pvc4)                FOLDER=crcns-pvc4 ;;
    mt1_norm_neutralbg)  FOLDER=crcns-mt1 ;;
    mt2)                 FOLDER=crcns-mt2 ;;
    mst_norm_neutralbg)  FOLDER=packlab-mst ;;
    *) echo "unknown dataset $DATASET"; exit 1 ;;
esac

mkdir -p "$DATA_ROOT" "$CKPT_ROOT" "$CACHE_ROOT"

# --- pull data + checkpoints from GCS (idempotent) ---
if [ ! -d "$DATA_ROOT/$FOLDER" ]; then
    echo ">>> fetching $FOLDER from GCS"
    gsutil -q cp "$BUCKET/data_derived/$FOLDER.zip" "/tmp/$FOLDER.zip"
    # use python zipfile to avoid depending on a system `unzip`
    "${PYTHON:-python3}" -c "import zipfile; zipfile.ZipFile('/tmp/$FOLDER.zip').extractall('$DATA_ROOT')"
    rm -f "/tmp/$FOLDER.zip"
fi
for ck in vjepa2_1_vitl_dist_vitG_384.pt midway-bdd-vit-b-ep300.pth; do
    [ -f "$CKPT_ROOT/$ck" ] || gsutil -q cp "$BUCKET/checkpoints/$ck" "$CKPT_ROOT/"
done

# Pull any existing feature cache for this (dataset, model) so fit jobs reuse it.
gsutil -m -q cp "$BUCKET/features/${FEATURES}_${DATASET}/*" "$CACHE_ROOT/" 2>/dev/null || true

AGGREGATOR="${AGGREGATOR:-average}"      # global token pool -> ~1GB cache/cell
VJEPA_PAD_T="${VJEPA_PAD_T:-16}"         # 10 -> 16 frames so T'=8 (ignored by Midway)
COMMON=(--exp_name "$EXP_NAME" --dataset "$DATASET" --features "$FEATURES"
        --data_root "$DATA_ROOT" --ckpt_root "$CKPT_ROOT" --cache_root "$CACHE_ROOT"
        --aggregator "$AGGREGATOR" --aggregator_sz 8 --pca 500 --method ridge
        --resize 112 --device "$DEVICE" --input_adapt "$INPUT_ADAPT"
        --vjepa_pad_t "$VJEPA_PAD_T")

if [ "$MODE" = "extract" ]; then
    "${PYTHON:-python3}" train_convex.py "${COMMON[@]}" --subset 0 --batch_size 8 --extract_only
    echo ">>> pushing feature cache to GCS"
    gsutil -m -q cp "$CACHE_ROOT"/* "$BUCKET/features/${FEATURES}_${DATASET}/"
else
    export WANDB_MODE="${WANDB_MODE:-offline}"
    "${PYTHON:-python3}" train_convex.py "${COMMON[@]}" --subset "$SUBSET" --batch_size 8 --save_predictions
    echo ">>> pushing results to GCS"
    gsutil -m -q cp -r wandb "$BUCKET/results/${FEATURES}_${DATASET}/subset${SUBSET}/" 2>/dev/null || true
fi
echo ">>> worker done: $MODE $DATASET $FEATURES $SUBSET"
