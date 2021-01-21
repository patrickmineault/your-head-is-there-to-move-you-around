#!/bin/bash
# set -e

#models=(gaborpyramid3d V1Net ShallowMonkeyNet_pvc1 ShallowMonkeyNet_pvc4 resnet18 MotionNet ShiftNet SlowFast_Slow SlowFast_Fast Slow I3D r3d_18 mc3_18 r2plus1d_18)
#models=(gaborpyramid3d V1Net ShallowMonkeyNet_pvc1 ShallowMonkeyNet_pvc4 resnet18 MotionNet ShiftNet SlowFast_Slow SlowFast_Fast Slow I3D r3d_18 mc3_18 r2plus1d_18)
models=(I3D r3d_18 mc3_18 r2plus1d_18)

for subset in s1 s2 s3;
do
    for model in "${models[@]}";
    do
        python train_convex.py \
            --exp_name V1Net \
            --dataset "vim2_deconv" \
            --features "$model" \
            --subset "$subset" \
            --batch_size 2 \
            --slowfast_root ../slowfast \
            --aggregator downsample \
            --aggregator_sz 8 \
            --pca 500 \
            --no_save \
            --cache_root /mnt/e/cache \
            --data_root /mnt/e/data_derived \
            --skip_existing
        rm -f /mnt/e/cache/*
    done
done

# #
# for subset in {0..24};
# do
#     python train_convex.py \
#         --exp_name V1Net \
#         --dataset "pvc4" \
#         --features "$model" \
#         --subset "$subset" \
#         --batch_size 8 \
#         --slowfast_root ../slowfast \
#         --aggregator downsample \
#         --aggregator_sz 8 \
#         --pca 500 \
#         --no_save \
#         --cache_root /mnt/e/cache \
#         --data_root /mnt/e/data_derived

#     # Clear cache.
#     rm /mnt/e/cache/*
# done

# for subset in {3..43};
# do
#     python train_convex.py \
#         --exp_name V1Net \
#         --dataset "mt2" \
#         --features "$model" \
#         --subset "$subset" \
#         --batch_size 8 \
#         --slowfast_root ../slowfast \
#         --aggregator downsample \
#         --aggregator_sz 8 \
#         --pca 500 \
#         --no_save \
#         --cache_root /mnt/e/cache \
#         --data_root /mnt/e/data_derived

#     # Clear cache.
#     rm /mnt/e/cache/*
# done

# for subset in {0..120};
# do
#     python train_convex.py \
#         --exp_name V1Net \
#         --dataset "v2" \
#         --features "$model" \
#         --subset "$subset" \
#         --batch_size 8 \
#         --slowfast_root ../slowfast \
#         --aggregator downsample \
#         --aggregator_sz 8 \
#         --pca 500 \
#         --no_save \
#         --cache_root /mnt/e/cache \
#         --data_root /mnt/e/data_derived

#     # Clear cache.
#     rm /mnt/e/cache/*
# done
