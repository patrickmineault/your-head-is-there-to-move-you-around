#!/bin/bash

for i in {0..24}
do
    python train_net.py --exp_name "_pvc4_pyramid_cell_${i}" \
        --single_cell "$i" \
        --learning_rate 3e-3 \
        --num_epochs 200 \
        --nfeats 16 \
        --warmup 1000 \
        --submodel gaborpyramid2d \
        --data_root /storage/crcns/ \
        --output_dir /storage/models/pvc4-pyramid \
        --ckpt_frequency 50000
done