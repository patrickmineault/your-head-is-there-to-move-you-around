#!/bin/bash


models=(cpc_01 cpc_02 airsim_04 gaborpyramid3d gaborpyramid3d_motionless MotionNet r3d_18 mc3_18 r2plus1d_18 I3D SlowFast_Fast)
data_root=/mnt/e/data_derived
slowfast_root=../slowfast
ckpt_root=./pretrained

for model in "${models[@]}";
do
    python train_heading.py \
        --exp_name cloudfit \
        --features "$model" \
        --data_root $data_root \
        --dataset airsim_batch2 \
        --batch_size 16 \
        --learning_rate 3e-3 \
        --softmax \
        --decoder center \
        --num_epochs 100 \
        --slowfast_root $slowfast_root \
        --ckpt_root $ckpt_root \
        --num_epochs 50
done