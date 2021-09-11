#!/bin/bash
set -e

# Somehow, the path is not correctly set - I don't totally get it
PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

case $DATASET in
    "crcns-pvc1")
        dataset_num=0
        size=21
        ;;
    "crcns-pvc4")
        dataset_num=1
        size=21
        ;;
    "crcns-mt1")
        dataset_num=2
        size=12
        ;;
    "crcns-mt2")
        dataset_num=3
        size=16
        ;;
    "packlab-mst")
        dataset_num=4
        size=21
        ;;
    *)
        echo "Unknown dataset"
        exit 0;
esac

echo "Free space info"
df -h

datasets=(pvc1-repeats pvc4 mt1_norm_neutralbg mt2 mst_norm_neutralbg)
max_cells=(22 24 83 43 35)
dataset=${datasets[$dataset_num]}
max_cell=${max_cells[$dataset_num]}

export AWS_RETRY_MODE=standard
export AWS_MAX_ATTEMPTS=3

pip install -r requirements.txt
aws s3 sync "s3://yourheadisthere/data_derived/$DATASET" "/data/data_derived/$DATASET"
aws s3 sync "s3://yourheadisthere/checkpoints" "/data/checkpoints"

# Not sure if actually necessary.
chown -R nobody:nogroup /data
chown -R nobody:nogroup /cache

ckpt_root=/data/checkpoints
data_root=/data/data_derived
cache_root=/cache
slowfast_root=../slowfast

# airsim_04 MotionNet
model=$MODEL
echo "$dataset" "$model"
for ((subset = 0; subset <= $max_cell; subset++))
do
    echo "Fitting cell $subset"
    python train_convex.py \
        --exp_name boosting_no_resize \
        --dataset "$dataset" \
        --features "$model" \
        --subset "$subset" \
        --batch_size 8 \
        --cache_root $cache_root \
        --ckpt_root $ckpt_root \
        --data_root $data_root \
        --slowfast_root $slowfast_root \
        --aggregator downsample_t \
        --aggregator_sz $size \
        --skip_existing \
        --subsample_layers \
        --autotune \
        --no_save \
        --save_predictions \
        --method boosting

    # Clear cache.
    rm -f $cache_root/*
done
