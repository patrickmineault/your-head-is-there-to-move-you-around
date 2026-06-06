#!/bin/bash
# Stage the 5 derived electrophysiology datasets + the two ViT checkpoints into GCS.
#
# Datasets are streamed S3 -> GCS (curl | gsutil cp -) so the ~46 GB never touches
# local disk. Checkpoints are uploaded from the local ./checkpoints dir.
#
# Usage: scripts/sync_datasets_to_gcs.sh [BUCKET]
set -euo pipefail

BUCKET="${1:-gs://xcorr-dev-motion-model}"
S3="https://yourheadisthere-data.s3.us-east-2.amazonaws.com/zips"
DATASETS=(crcns-pvc1 crcns-pvc4 crcns-mt1 crcns-mt2 packlab-mst)

echo ">>> Staging datasets to ${BUCKET}/data_derived/"
for d in "${DATASETS[@]}"; do
    dst="${BUCKET}/data_derived/${d}.zip"
    if gsutil -q stat "$dst" 2>/dev/null; then
        echo "  [skip] $d.zip already in GCS"
        continue
    fi
    echo "  [stream] $d.zip -> $dst"
    curl -fsSL "${S3}/${d}.zip" | gsutil cp - "$dst"
done

echo ">>> Uploading checkpoints to ${BUCKET}/checkpoints/"
CKPT_DIR="${CKPT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)/checkpoints}"
gsutil -m cp -n \
    "${CKPT_DIR}/vjepa2_1_vitl_dist_vitG_384.pt" \
    "${CKPT_DIR}/midway-bdd-vit-b-ep300.pth" \
    "${BUCKET}/checkpoints/"

echo ">>> Done. Contents:"
gsutil ls -l "${BUCKET}/data_derived/" "${BUCKET}/checkpoints/"
