#!/bin/bash
set -e
ckpt_root=./pretrained;
data_root=/mnt/e/data_derived;
cache_root=/mnt/e/cache;
slowfast_root=../slowfast;

models=(airsim_04 MotionNet) 
datasets=(pvc1-repeats pvc4 mt1_norm_neutralbg mt2 mst_norm_neutralbg)
max_cells=(22 24 83 43 35)
# Whatever is the largest subsampling that fits into main memory for that dataset.
# We used 64GB VMs.
subsamplings=(21 21 12 16 21)  

#.66X, 1X, 1.5X
szs=(74 112 168)  

# Fit with resizing, boosting with downsampling
for dataset_num in {0..4};
do
    dataset=${datasets[$dataset_num]}
    max_cell=${max_cells[$dataset_num]}
    for sz in "${szs[@]}";
    do
        for model in "${models[@]}";
        do
            echo "$dataset" "$model"
            for ((subset = 0; subset <= max_cell; subset++))
            do
                python train_convex.py \
                    --exp_name revision \
                    --dataset "$dataset" \
                    --features "$model" \
                    --subset "$subset" \
                    --batch_size 8 \
                    --cache_root $cache_root \
                    --ckpt_root $ckpt_root \
                    --data_root $data_root \
                    --slowfast_root $slowfast_root \
                    --aggregator downsample \
                    --aggregator_sz 8 \
                    --skip_existing \
                    --subsample_layers \
                    --autotune \
                    --no_save \
                    --save_predictions \
                    --method boosting \
                    --resize "$sz"
                # Clear cache.
                rm -f $cache_root/*
            done
        done
    done
done

# Fit with resizing, boosting with subsampling
for dataset_num in {0..4};
do
    dataset=${datasets[$dataset_num]}
    max_cell=${max_cells[$dataset_num]}
    subsampling=${subsamplings[$dataset_num]}
    for sz in "${szs[@]}";
    do
        for model in "${models[@]}";
        do
            echo "$dataset" "$model"
            for ((subset = 0; subset <= max_cell; subset++))
            do
                python train_convex.py \
                    --exp_name revision \
                    --dataset "$dataset" \
                    --features "$model" \
                    --subset "$subset" \
                    --batch_size 8 \
                    --cache_root $cache_root \
                    --ckpt_root $ckpt_root \
                    --data_root $data_root \
                    --slowfast_root $slowfast_root \
                    --aggregator downsample_t \
                    --aggregator_sz "$subsampling" \
                    --skip_existing \
                    --subsample_layers \
                    --autotune \
                    --no_save \
                    --save_predictions \
                    --method boosting \
                    --resize "$sz"
                # Clear cache.
                rm -f $cache_root/*
            done
        done
    done
done

# Fit 
for dataset_num in {0..4};
do
    dataset=${datasets[$dataset_num]}
    max_cell=${max_cells[$dataset_num]}
    for sz in "${szs[@]}";
    do
        for model in "${models[@]}";
        do
            echo "$dataset" "$model"
            for ((subset = 0; subset <= max_cell; subset++))
            do
                python train_convex.py \
                    --exp_name revision_resize \
                    --dataset "$dataset" \
                    --features "$model" \
                    --subset "$subset" \
                    --batch_size 8 \
                    --cache_root $cache_root \
                    --ckpt_root $ckpt_root \
                    --data_root $data_root \
                    --slowfast_root $slowfast_root \
                    --aggregator downsample \
                    --aggregator_sz 8 \
                    --skip_existing \
                    --subsample_layers \
                    --autotune \
                    --no_save \
                    --save_predictions \
                    --method ridge \
                    --pca 500 \
                    --resize "$sz"
                # Clear cache.
                rm -f $cache_root/*
            done
        done
    done
done
