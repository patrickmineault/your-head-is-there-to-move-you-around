#!/bin/bash
set -e

size=8
ckpt_root=./pretrained;
data_root=/mnt/e/data_derived;
cache_root=/mnt/e/cache;
slowfast_root=../slowfast;

models=(MotionNet airsim_04) 
datasets=(mt2)
max_cells=(43)

for dataset_num in {0..0};
do
    for model in "${models[@]}";
    do
        dataset=${datasets[$dataset_num]}
        max_cell=${max_cells[$dataset_num]}

        echo "$dataset" "$model"
        for ((subset = 0; subset <= $max_cell; subset++))
        do
            python train_convex.py \
                --exp_name revision_resize_boost \
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
                --method boosting \
                --resize 168
            # Clear cache.
            rm -f $cache_root/*
        done
    done
done