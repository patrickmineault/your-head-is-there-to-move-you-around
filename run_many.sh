#!/bin/bash

models=(gaborpyramid3d r3d_18 mc3_18 r2plus1d_18 vgg19 r3d_18 r3d_18)
max_layer=(4 16 16 16 15 16 16)
width=(112 112 112 112 112 56 28)
for subject in s1 s2 s3;
    do
    for i in "${!models[@]}";
    do
        for layer in $(seq 0 1 "${max_layer[$i]}");
        do
            python train_fmri_convex.py \
                --exp_name 20201226 \
                --layer "$layer" \
                --features "${models[$i]}" \
                --subject "$subject" \
                --width "${width[$i]}"
        done
    done
done