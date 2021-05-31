#!/bin/bash
set -e

size=8
ckpt_root=./pretrained;
data_root=/mnt/e/data_derived;
cache_root=/mnt/e/cache;
slowfast_root=../slowfast;

models=(cpc_01 cpc_02 airsim_04 gaborpyramid3d gaborpyramid3d_motionless MotionNet SlowFast_Fast I3D r3d_18 mc3_18 r2plus1d_18)
datasets=(mt1_norm_neutralbg mt2 pvc1-repeats pvc4 mst_norm_neutralbg)
max_cells=(83 43 22 24 35)

for dataset_num in {0..4};
do
    dataset=${datasets[$dataset_num]}
    max_cell=${max_cells[$dataset_num]}
    for model in "${models[@]}";
    do
        echo "$dataset" "$model"
        for ((subset = 0; subset <= $max_cell; subset++))
        do
            python train_convex.py \
                --exp_name fit_all \
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
                --method ridge \
                --pca 500
            # Clear cache.
            rm -f $cache_root/*
        done
    done
done




dataset=mst_norm_neutralbg
models=(SlowFast_Fast)
for model in "${models[@]}";
do
    for subset in {0..35};
    do
        python train_convex.py \
            --exp_name 20210524 \
            --dataset "$dataset" \
            --features "$model" \
            --subset "$subset" \
            --batch_size 4 \
            --slowfast_root $slowfast_root \
            --ckpt_root $ckpt_root \
            --aggregator downsample \
            --aggregator_sz 8 \
            --pca 500 \
            --no_save \
            --cache_root $cache_root \
            --data_root $data_root \
            --autotune \
            --subsample_layers \
            --skip_existing
        # Clear cache.
        rm -f $cache_root/*
    done
done


# dataset=pvc4
# models=(airsim_04)
# for model in "${models[@]}";
# do
#     for subset in {0..24};
#     do
#         python train_convex.py \
#             --exp_name Benchmark \
#             --dataset "$dataset" \
#             --features "$model" \
#             --subset "$subset" \
#             --batch_size 4 \
#             --slowfast_root $slowfast_root \
#             --ckpt_root $ckpt_root \
#             --aggregator downsample \
#             --aggregator_sz 8 \
#             --pca 500 \
#             --no_save \
#             --cache_root $cache_root \
#             --data_root $data_root \
#             --autotune \
#             --skip_existing
#         # Clear cache.
#         rm -f $cache_root/*
#     done
# done



# models=(airsim_04 
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
# models=(cpc_01 cpc_02)
# airsim_04 gaborpyramid3d gaborpyramid3d_motionless MotionNet ShiftNet Slow I3D r3d_18 mc3_18 r2plus1d_18)
# size=8
# for dataset in mst_norm_airsim mst_norm_neutralbg;
# do
#     for model in "${models[@]}";
#     do
#         for subset in {0..35};
#         do
#             python train_convex.py \
#                 --exp_name 20210324 \
#                 --dataset "$dataset" \
#                 --features "$model" \
#                 --subset "$subset" \
#                 --batch_size 8 \
#                 --cache_root $cache_root \
#                 --ckpt_root $ckpt_root \
#                 --data_root $data_root \
#                 --slowfast_root $slowfast_root \
#                 --aggregator downsample \
#                 --aggregator_sz $size \
#                 --skip_existing \
#                 --subsample_layers \
#                 --autotune \
#                 --no_save \
#                 --save_predictions \
#                 --method ridge \
#                 --pca 500 \
#             # Clear cache.
#             rm -f $cache_root/*
#         done
#     done
# done


# models=(gaborpyramid3d gaborpyramid3d_motionless cpc_01 cpc_02 airsim_04 MotionNet SlowFast_Fast Slow I3D r3d_18 mc3_18 r2plus1d_18)
# size=8
# dataset="dorsal_norm_neutralbg"
# for model in "${models[@]}";
# do
#     for subset in {0..22};
#     do
#         python train_convex.py \
#             --exp_name 20210427 \
#             --dataset "$dataset" \
#             --features "$model" \
#             --subset "$subset" \
#             --batch_size 8 \
#             --cache_root $cache_root \
#             --ckpt_root $ckpt_root \
#             --data_root $data_root \
#             --slowfast_root $slowfast_root \
#             --aggregator downsample \
#             --aggregator_sz $size \
#             --skip_existing \
#             --subsample_layers \
#             --autotune \
#             --no_save \
#             --save_predictions \
#             --method ridge \
#             --pca 500 \
#         # Clear cache.
#         rm -f $cache_root/*
#     done
# done

models=(cpc_01 cpc_02 airsim_04 gaborpyramid3d gaborpyramid3d_motionless MotionNet SlowFast_Fast Slow I3D r3d_18 mc3_18 r2plus1d_18)

size=8
dataset="mt1_norm_neutralbg"
for model in "${models[@]}";
do
    for subset in {0..83};
    do
        python train_convex.py \
            --exp_name 20210503 \
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
            --method ridge \
            --pca 500 \
        # Clear cache.
        rm -f $cache_root/*
    done
done
