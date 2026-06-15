#!/bin/bash
# Worker entrypoint for one unit of work (GCE GPU VM / GCP Batch task / local).
# A work unit is one of:
#   MODE=extract  -> build the feature cache for (DATASET, FEATURES) then push it
#   MODE=loop     -> fit EVERY neuron of (DATASET, FEATURES): pull data+ckpt once,
#                    loop subsets, skip ones already in GCS  [recommended: amortizes
#                    VM provision / deps install / data pull across all neurons]
#   MODE=fit      -> ridge-fit a single neuron (DATASET, FEATURES, SUBSET)
#
# Work unit comes from explicit env vars, or, if MANIFEST(_GCS) is set, from the
# (0-based) line ${BATCH_TASK_INDEX} of a TSV "MODE DATASET FEATURES SUBSET".
set -euo pipefail

BUCKET="${BUCKET:-gs://xcorr-dev-motion-model}"
DATA_ROOT="${DATA_ROOT:-/data/data_derived}"
CKPT_ROOT="${CKPT_ROOT:-/data/checkpoints}"
CACHE_ROOT="${CACHE_ROOT:-/cache}"
DEVICE="${DEVICE:-cuda}"
INPUT_ADAPT="${INPUT_ADAPT:-resize}"
EXP_NAME="${EXP_NAME:-vjepa_midway_fit}"
AGGREGATOR="${AGGREGATOR:-average}"      # global token pool -> ~0.7GB cache/cell
VJEPA_PAD_T="${VJEPA_PAD_T:-16}"         # 10 -> 16 frames so T'=8 (ignored by Midway)
BATCH_SIZE="${BATCH_SIZE:-16}"           # fp16 + 256px V-JEPA -> 16 fits an L4 comfortably
CHPROJ_DIM="${CHPROJ_DIM:-64}"           # for aggregator=downsample_chproj (C -> this)

# neuron count per dataset (max subset index, inclusive) -- matches run_remote.sh
declare -A MAXCELLS=(
    [pvc1-repeats]=22 [pvc4]=24 [mt1_norm_neutralbg]=83 [mt2]=43 [mst_norm_neutralbg]=35
)

if [ -n "${MANIFEST_GCS:-}" ]; then
    gsutil -q cp "$MANIFEST_GCS" /tmp/manifest.tsv && MANIFEST=/tmp/manifest.tsv
fi
if [ -n "${MANIFEST:-}" ]; then
    line=$(sed -n "$((${BATCH_TASK_INDEX:-0} + 1))p" "$MANIFEST")
    read -r MODE DATASET FEATURES SUBSET <<< "$line"
fi
: "${MODE:?}"; : "${DATASET:?}"; : "${FEATURES:?}"; SUBSET="${SUBSET:--}"

case "$DATASET" in
    pvc1-repeats)        FOLDER=crcns-pvc1 ;;
    pvc4)                FOLDER=crcns-pvc4 ;;
    mt1_norm_neutralbg)  FOLDER=crcns-mt1 ;;
    mt2)                 FOLDER=crcns-mt2 ;;
    mst_norm_neutralbg)  FOLDER=packlab-mst ;;
    *) echo "unknown dataset $DATASET"; exit 1 ;;
esac
case "$FEATURES" in
    vjepa2_1_vitl)   ck=vjepa2_1_vitl_dist_vitG_384.pt ;;
    midway_bdd_vitb) ck=midway-bdd-vit-b-ep300.pth ;;
    *) echo "unknown features $FEATURES"; exit 1 ;;
esac

mkdir -p "$DATA_ROOT" "$CKPT_ROOT" "$CACHE_ROOT"

# --- pull data + checkpoint ONCE per task (idempotent) ---
if [ ! -d "$DATA_ROOT/$FOLDER" ]; then
    echo ">>> fetching $FOLDER from GCS"
    gsutil -q cp "$BUCKET/data_derived/$FOLDER.zip" "/tmp/$FOLDER.zip"
    "${PYTHON:-python3}" -c "import zipfile; zipfile.ZipFile('/tmp/$FOLDER.zip').extractall('$DATA_ROOT')"
    rm -f "/tmp/$FOLDER.zip"
fi
[ -f "$CKPT_ROOT/$ck" ] || gsutil -q cp "$BUCKET/checkpoints/$ck" "$CKPT_ROOT/"

COMMON=(--exp_name "$EXP_NAME" --dataset "$DATASET" --features "$FEATURES"
        --data_root "$DATA_ROOT" --ckpt_root "$CKPT_ROOT" --cache_root "$CACHE_ROOT"
        --aggregator "$AGGREGATOR" --aggregator_sz 8 --pca 500 --method ridge
        --resize 112 --device "$DEVICE" --input_adapt "$INPUT_ADAPT"
        --vjepa_pad_t "$VJEPA_PAD_T" --chproj_dim "$CHPROJ_DIM")

# Online W&B if a key is present (injected from Secret Manager by Batch), else offline.
if [ -n "${WANDB_API_KEY:-}" ]; then export WANDB_MODE=online; else export WANDB_MODE=offline; fi

run_fit() {  # $1 = subset; skips if already in GCS, clears per-cell cache after
    local s="$1"
    local rpath="$BUCKET/results/${EXP_NAME}/${FEATURES}_${DATASET}/subset${s}"
    if gsutil -q ls "$rpath/**/results.pkl" >/dev/null 2>&1; then
        echo ">>> subset $s already done, skipping"; return 0
    fi
    rm -rf wandb
    "${PYTHON:-python3}" train_convex.py "${COMMON[@]}" --subset "$s" --batch_size "$BATCH_SIZE" --save_predictions
    gsutil -m -q cp -r wandb "$rpath/" 2>/dev/null || true
    # each cell's features differ (per-cell stimuli) -> drop its cache to bound disk
    rm -f "$CACHE_ROOT/${FEATURES}_"*"_${DATASET}_${s}_"*.h5 2>/dev/null || true
    echo ">>> subset $s done"
}

case "$MODE" in
    extract)
        "${PYTHON:-python3}" train_convex.py "${COMMON[@]}" --subset 0 --batch_size "$BATCH_SIZE" --extract_only
        gsutil -m -q cp "$CACHE_ROOT"/* "$BUCKET/features/${FEATURES}_${DATASET}/" ;;
    loop)
        max="${MAXCELLS[$DATASET]}"
        echo ">>> looping $DATASET/$FEATURES neurons 0..$max"
        for s in $(seq 0 "$max"); do run_fit "$s"; done ;;
    *)
        run_fit "$SUBSET" ;;
esac
echo ">>> worker done: $MODE $DATASET $FEATURES"
