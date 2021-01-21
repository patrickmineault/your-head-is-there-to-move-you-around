#!/bin/bash
# set -e

# TODO: figure out what to do about SlowFast_Slow and SlowFast_Fast

#models=(gaborpyramid3d ShallowMonkeyNet_pvc1 ShallowMonkeyNet_pvc4 resnet18 MotionNet ShiftNet SlowFast_Slow SlowFast_Fast Slow I3D r3d_18 mc3_18 r2plus1d_18 vgg19)
#models=(ShiftNet r3d_18 mc3_18 r2plus1d_18 SlowFast_Slow SlowFast_Fast I3D Slow)
#models=(ShallowMonkeyNet_pvc1)
#models=(resnet18)
#for model in "${models[@]}";
#do

# model="r3d_18"
# subset="s1"

# python train_convex.py \
#     --exp_name V1Net \
#     --dataset "vim2" \
#     --features "$model" \
#     --subset "$subset" \
#     --batch_size 8 \
#     --slowfast_root ../slowfast \
#     --aggregator downsample \
#     --aggregator_sz 8 \
#     --pca 500 \
#     --no_save \
#     --cache_root /mnt/e/cache \
#     --data_root /mnt/e/data_derived

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

# model=gaborpyramid3d

# for subset in {0..43};
# do
#     python train_convex.py \
#         --exp_name Benchmark \
#         --dataset "mt2" \
#         --features "$model" \
#         --subset "$subset" \
#         --batch_size 64 \
#         --slowfast_root ../slowfast \
#         --aggregator downsample \
#         --aggregator_sz 8 \
#         --pca 500 \
#         --no_save \
#         --cache_root /mnt/e/cache \
#         --data_root /mnt/e/data_derived \
#         --skip_existing
#     # Clear cache.
#     rm -f /mnt/e/cache/*
# done


# model=gaborpyramid3d_motionless
# for subset in {0..120};
# do
#     python train_convex.py \
#         --exp_name Benchmark \
#         --dataset "v2" \
#         --features "$model" \
#         --subset "$subset" \
#         --batch_size 64 \
#         --slowfast_root ../slowfast \
#         --aggregator downsample \
#         --aggregator_sz 8 \
#         --pca 500 \
#         --no_save \
#         --cache_root /mnt/e/cache \
#         --data_root /mnt/e/data_derived \
#         --skip_existing
#         # Clear cache.
#     rm -f /mnt/e/cache/*
# done

# model=gaborpyramid3d
# for subset in {0..24};
# do
#     python train_convex.py \
#         --exp_name Benchmark \
#         --dataset "pvc4" \
#         --features "$model" \
#         --subset "$subset" \
#         --batch_size 64 \
#         --slowfast_root ../slowfast \
#         --aggregator downsample \
#         --aggregator_sz 8 \
#         --pca 500 \
#         --no_save \
#         --cache_root /mnt/e/cache \
#         --data_root /mnt/e/data_derived \
#         --skip_existing

#     # Clear cache.
#     rm /mnt/e/cache/*
# done

# for subset in {0..123};
# do
#     python train_convex.py \
#         --exp_name Benchmark \
#         --dataset "pvc1" \
#         --features "$model" \
#         --subset "$subset" \
#         --batch_size 64 \
#         --slowfast_root ../slowfast \
#         --aggregator downsample \
#         --aggregator_sz 8 \
#         --pca 500 \
#         --no_save \
#         --cache_root /mnt/e/cache \
#         --data_root /mnt/e/data_derived \
#         --skip_existing

#     # Clear cache.
#     rm /mnt/e/cache/*
# done

for subset in {0..123};
do
    for model in gaborpyramid3d gaborpyramid3d_motionless;
    do
        python train_convex.py \
            --exp_name Benchmark \
            --dataset "pvc1" \
            --features "$model" \
            --subset "$subset" \
            --batch_size 64 \
            --slowfast_root ../slowfast \
            --aggregator average \
            --aggregator_sz 8 \
            --no_save \
            --cache_root /mnt/e/cache \
            --data_root /mnt/e/data_derived \
            --skip_existing \
            --consolidated

        # Clear cache.
        rm /mnt/e/cache/*
    done
done