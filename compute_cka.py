from modelzoo import xception, separable_net, gabor_pyramid, dorsalnet, decoder
from loaders import airsim
from models import extract_subnet_dict

import argparse
import datetime
import itertools
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import numpy as np
import torch
from torch import nn
from torch import optim

import torch.autograd.profiler as profiler
from torchvision import transforms
import torchvision.models as models
import torch.nn.functional as F

import models

from transforms import ThreedGaussianBlur, ThreedExposure

import train_heading

import wandb

from models import preprocess_data
from research_code.cka_step4 import cka, multi_cka

from models import get_aggregator


def main(args):
    print("Main")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("No CUDA! Sad!")

    (
        trainset,
        tuneset,
        reportset,
        train_transform,
        eval_transform,
        start_sz,
    ) = train_heading.get_dataset(args)

    reportloader = torch.utils.data.DataLoader(
        reportset, batch_size=args.batch_size, shuffle=True, pin_memory=True
    )

    print("Init models")

    subnet, activations, metadata = train_heading.get_subnet(args, start_sz)
    aggregator = get_aggregator(metadata, args)

    subnet.to(device=device)
    subnet.eval()

    # Use the report fold as a means of evaluating the CKA alignments.
    layer_responses = []
    for layer_num, layer_name in enumerate(metadata["layers"].keys()):
        args.layer = layer_num  # For backwards compatibility
        args.layer_name = layer_name
        args.subset = 0
        X, Y = preprocess_data(
            reportloader, subnet, aggregator, activations, metadata, args
        )

        # Pick a time and position slice in the middle.
        X = X.reshape((X.shape[0], -1, 4, args.aggregator_sz, args.aggregator_sz))
        X = X[:, :, 2, args.aggregator_sz // 2, args.aggregator_sz // 2].reshape(
            (X.shape[0], -1)
        )
        layer_responses.append(X)

    alignments = multi_cka(layer_responses)
    np.save("cka.npy", alignments)


if __name__ == "__main__":
    desc = "Measure CKA between layers of a deep neural net"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--features",
        default="gaborpyramid3d",
        type=str,
        help="Sub-model type (currently, either xception2d, gaborpyramid2d, gaborpyramid3d",
    )
    parser.add_argument(
        "--dataset", default="airsim", help="Dataset (currently airsim only)"
    )
    parser.add_argument("--data_root", default="./data_derived", help="Data path")
    parser.add_argument("--ckpt_root", default="./checkpoints", help="Data path")
    parser.add_argument(
        "--output_dir", default="./models", help="Output path for models"
    )
    parser.add_argument(
        "--softmax",
        default=False,
        help="Use softmax objective rather than regression",
        action="store_true",
    )
    parser.add_argument(
        "--slowfast_root", default="", help="Path where SlowFast is installed"
    )
    parser.add_argument(
        "--subsample_layers",
        default=False,
        help="Subsample layers (saves disk space & mem)",
        action="store_true",
    )
    parser.add_argument(
        "--cache_root", default="./cache", help="Precomputed cache path"
    )
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
    parser.add_argument(
        "--aggregator",
        default="average",
        type=str,
        help="What kind of aggregator to use",
    )
    parser.add_argument(
        "--aggregator_sz",
        default=8,
        type=int,
        help="The size the aggregator will be used with",
    )

    args = parser.parse_args()
    main(args)
