#!/bin/bash
set -e

# TODO: figure out what to do about SlowFast_Slow and SlowFast_Fast
models=(gaborpyramid3d ShallowMonkeyNet_pvc1 ShallowMonkeyNet_pvc4 resnet18 MotionNet ShiftNet Slow I3D r3d_18 mc3_18 r2plus1d_18)
for model in "${models[@]}";
do
    for subset in {0..43};
    do
        python train_convex.py \
            --exp_name 20210109 \
            --dataset "mt2" \
            --features "$model" \
            --subset "$subset" \
            --batch_size 8 \
            --ckpt_root /storage/checkpoints \
            --data_root /storage/data_derived \
            --slowfast_root ../slowfast \
            --aggregator downsample \
            --aggregator_sz 8 \
            --pca 500 \
            --no_save
        # Clear cache.
        rm cache/*
    done
done