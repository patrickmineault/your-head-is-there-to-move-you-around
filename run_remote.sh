#!/bin/bash
set -e

# There's something about the indirection where the environment variables don't get forwarded, I don't totally get it.
PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

pip install -r requirements.txt
aws s3 sync s3://yourheadisthere/ /data

ls -al /data
ls -al /data/checkpoints
ls -al /data/data_derived
ls -al /data/data_derived/crcns-mt2

size=8
ckpt_root=/data/checkpoints
data_root=/data/data_derived
cache_root=/cache
slowfast_root=../slowfast

models=(airsim_04 MotionNet)
datasets=(mt2)
max_cells=(43)

for dataset_num in {0..0};
do
    dataset=${datasets[$dataset_num]}
    max_cell=${max_cells[$dataset_num]}
    for model in "${models[@]}";
    do
        echo "$dataset" "$model"
        for ((subset = 0; subset <= $max_cell; subset++))
        do
            python train_convex.py \
                --exp_name mt_boosting_revision \
                --dataset "$dataset" \
                --features "$model" \
                --subset "$subset" \
                --batch_size 8 \
                --cache_root $cache_root \
                --ckpt_root $ckpt_root \
                --data_root $data_root \
                --slowfast_root $slowfast_root \
                --aggregator downsample \
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
    done
done
