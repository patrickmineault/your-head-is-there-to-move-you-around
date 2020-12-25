#!/bin/bash

python train_fmri_convex.py --exp_name gaborpyramid3d --target_layer layer1 --features gaborpyramid3d
for i in {1..17}
    do
        python train_fmri_convex.py --exp_name "r3d_18_${i}" --target_layer "layer${i}" --features r3d_18
done