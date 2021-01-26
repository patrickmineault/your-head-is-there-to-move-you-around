#!/bin/bash
set -e

# TODO: figure out what to do about SlowFast_Slow and SlowFast_Fast Slow I3D
# models=(gaborpyramid3d r3d_18 ShallowMonkeyNet_pvc1 V1Net)
if [ "$1" == "resnet" ]; then
    # there's something broken about ShiftNet right now
    models=(ShiftNet mc3_18 r2plus1d_18 resnet18)
    size=8
elif [ "$1" == "slowfast" ]; then
    # Slow, , SlowFast_Fast broken
    # SlowFast_Slow will stay broken for a bit
    # Slow is a huge model so it's very slow.
    models=(SlowFast_Fast I3D)
    size=8
else
    echo "Unknown type!"
    exit 1
fi

for model in "${models[@]}";
do
    for subset in {0..24};
    do
        python train_convex.py \
            --exp_name 20210109 \
            --dataset "pvc4" \
            --features "$model" \
            --subset "$subset" \
            --batch_size 8 \
            --cache_root ./cache \
            --ckpt_root /storage/checkpoints \
            --data_root /storage/data_derived \
            --slowfast_root /workspace/slowfast \
            --aggregator downsample \
            --aggregator_sz $size \
            --pca 500 \
            --no_save \
            --skip_existing \
            --subsample_layers \
            --autotune

        # Clear cache.
        rm -f ./cache/*
    done
done