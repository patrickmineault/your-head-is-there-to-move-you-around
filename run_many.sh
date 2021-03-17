#!/bin/bash
set -e

# TODO: figure out what to do about SlowFast_Slow and SlowFast_Fast

#models=(gaborpyramid3d gaborpyramid3d_motionless airsim_04 MotionNet ShiftNet SlowFast_Fast Slow I3D r3d_18 mc3_18 r2plus1d_18 cpc)
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

# model=dorsalnet
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
#     rm -f /mnt/e/cache/*
# done

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

# model=dorsalnet
# for subset in s1;
# do
#     python train_convex.py \
#         --exp_name V1Net \
#         --dataset "vim2_deconv" \
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
#         --aggregator average \
#         --aggregator_sz 8 \
#         --no_save \
#         --cache_root /mnt/e/cache \
#         --data_root /mnt/e/data_derived \
#         --skip_existing \
#         --consolidated

#     # Clear cache.
#     rm -f /mnt/e/cache/*
# done

# model=airsim_02
# for subset in {0..24};
# do
#     python train_convex.py \
#         --exp_name Benchmark \
#         --dataset "pvc4" \
#         --features "$model" \
#         --subset "$subset" \
#         --batch_size 4 \
#         --slowfast_root ../slowfast \
#         --aggregator downsample \
#         --aggregator_sz 8 \
#         --pca 500 \
#         --no_save \
#         --cache_root /mnt/e/cache \
#         --data_root /mnt/e/data_derived \
#         --autotune \
#         --skip_existing
#     # Clear cache.
#     rm -f /mnt/e/cache/*
# done

# 
# for subset in s1 s2 s3;
# do
#     python train_convex.py \
#         --exp_name Benchmark \
#         --dataset "vim2" \
#         --features "$model" \
#         --subset "$subset" \
#         --batch_size 4 \
#         --slowfast_root ../slowfast \
#         --aggregator downsample \
#         --aggregator_sz 8 \
#         --pca 500 \
#         --no_save \
#         --cache_root /mnt/e/cache \
#         --data_root /mnt/e/data_derived \
#         --autotune \
#         --skip_existing
#     # Clear cache.
#     rm -f /mnt/e/cache/*
# done

ckpt_root=./pretrained;
data_root=/mnt/e/data_derived;
cache_root=/mnt/e/cache;
slowfast_root=../slowfast;

# model=airsim_04
# for subset in {0..24};
# do
#     python train_convex.py \
#         --exp_name Benchmark \
#         --dataset "pvc4" \
#         --features "$model" \
#         --subset "$subset" \
#         --batch_size 4 \
#         --slowfast_root $slowfast_root \
#         --ckpt_root $ckpt_root \
#         --aggregator downsample \
#         --aggregator_sz 8 \
#         --pca 500 \
#         --no_save \
#         --cache_root $cache_root \
#         --data_root $data_root \
#         --autotune \
#         --skip_existing
#     # Clear cache.
#     rm -f $cache_root/*
# done

#dataset=mst_norm_airsim  # mst_norm_neutralbg
#dataset=mst_norm_cpc
#models=(airsim_04)
models=(airsim_04 MotionNet gaborpyramid3d)
# airsim_04 gaborpyramid3d gaborpyramid3d_motionless MotionNet ShiftNet Slow I3D r3d_18 mc3_18 r2plus1d_18)
size=8
for dataset in mst_norm_neutralbg;
do
    for model in "${models[@]}";
    do
        for subset in {0..35};
        do
            python train_convex.py \
                --exp_name 20210305 \
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
            #/--pca 500 \
            # Clear cache.
            rm -f $cache_root/*
        done
    done
done