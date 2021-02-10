#!/bin/bash
set -e
models=(gaborpyramid3d_motionless gaborpyramid3d airsim_02 ShallowMonkeyNet_pvc1 ShallowMonkeyNet_pvc4 MotionNet ShiftNet SlowFast_Slow SlowFast_Fast I3D r3d_18 mc3_18 r2plus1d_18)

for model in "${models[@]}";
do
    python compare_reps.py \
        --exp_name st_v3a \
        --dataset "st-v3a" \
        --features "$model" \
        --batch_size 4 \
        --slowfast_root ../slowfast \
        --cache_root /mnt/e/cache \
        --data_root /mnt/e/data_derived \
        --skip_existing
    # Clear cache.
    rm -f /mnt/e/cache/*
done



# --skip_existing
