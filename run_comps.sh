#!/bin/bash
set -e
models=(airsim_03 airsim_04)

# for model in "${models[@]}";
# do
#     python compare_reps.py \
#         --exp_name st_v3a \
#         --dataset "st-v3a" \
#         --features "$model" \
#         --batch_size 4 \
#         --slowfast_root ../slowfast \
#         --cache_root /mnt/e/cache \
#         --data_root /mnt/e/data_derived \
#         --skip_existing
#     # Clear cache.
#     rm -f /mnt/e/cache/*
# done

ckpt_root=./pretrained;
data_root=/mnt/e/data_derived;
cache_root=/mnt/e/cache;
# model=airsim_03;

# for subset in {0..43};
# do
#     python train_convex.py \
#         --exp_name 20210109 \
#         --dataset "mt2" \
#         --features "$model" \
#         --subset "$subset" \
#         --batch_size 8 \
#         --cache_root $cache_root \
#         --ckpt_root $ckpt_root \
#         --data_root $data_root \
#         --slowfast_root /workspace/slowfast \
#         --aggregator downsample \
#         --aggregator_sz 8 \
#         --pca 500 \
#         --no_save \
#         --subsample_layers \
#         --autotune

#     # Clear cache.
#     rm -f $cache_root/*
# done

model=airsim_04;

# python compare_reps.py \
#     --exp_name st_mst \
#     --dataset "st-mst" \
#     --features "$model" \
#     --batch_size 4 \
#     --slowfast_root ../slowfast \
#     --cache_root $cache_root \
#     --data_root $data_root
# # Clear cache.
# rm -f $cache_root/*

for subset in {0..43};
do
    python train_convex.py \
        --exp_name 20210109 \
        --dataset "mt2" \
        --features "$model" \
        --subset "$subset" \
        --batch_size 8 \
        --cache_root $cache_root \
        --ckpt_root $ckpt_root \
        --data_root $data_root \
        --slowfast_root /workspace/slowfast \
        --aggregator downsample \
        --aggregator_sz 8 \
        --pca 500 \
        --no_save \
        --subsample_layers \
        --autotune

    # Clear cache.
    rm -f $cache_root/*
done


# --skip_existing
