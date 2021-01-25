#!/bin/bash
set -e

# TODO: figure out what to do about SlowFast_Slow and SlowFast_Fast Slow I3D
# models=(gaborpyramid3d r3d_18 ShallowMonkeyNet_pvc1 V1Net)
if [ "$1" == "resnet" ]; then
    models=(MotionNet ShiftNet resnet18 mc3_18 r2plus1d_18)
    size=8
elif [ "$1" == "slowfast" ]; then
    models=(Slow SlowFast_Fast SlowFast_Slow I3D)
    size=4
else
    echo "Unknown type!"
    exit 1
fi


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
            --cache_root ./cache \
            --ckpt_root /storage/checkpoints \
            --data_root /storage/data_derived \
            --slowfast_root /workspace/slowfast \
            --aggregator downsample \
            --aggregator_sz $size \
            --pca 500 \
            --no_save \
            --skip_existing \
            --subsample_layers

        # Clear cache.
        rm -f ./cache/*
    done
done