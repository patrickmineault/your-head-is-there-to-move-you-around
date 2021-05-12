#!/bin/bash


if [ "$1" == "cpc" ]; then
    models=(cpc_01 cpc_02)
elif [ "$1" == "baselines" ]; then
    models=(airsim_04 gaborpyramid3d gaborpyramid3d_motionless MotionNet)
elif [ "$1" == "r3d" ]; then
    models=(r3d_18 mc3_18 r2plus1d_18)
elif [ "$1" == "slowfast" ]; then
    models=(SlowFast_Fast I3D)
else
    echo "Unknown type!"
    exit 1
fi


if [ "$2" == "local" ]; then
    data_root=/mnt/e/data_derived
    slowfast_root=../slowfast
    ckpt_root=./pretrained
elif [ "$2" == "remote" ]; then
    data_root=/storage/data_derived
    slowfast_root=/workspace/slowfast
    ckpt_root=/storage/checkpoints
else
    echo "Unknown target!"
    exit 1
fi


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