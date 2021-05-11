from modelzoo import xception, separable_net, gabor_pyramid, monkeynet, decoder
from loaders import airsim
from fmri_models import extract_subnet_dict

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
from torch.utils.tensorboard import SummaryWriter
import torch.autograd.profiler as profiler
from torchvision import transforms
import torchvision.models as models
import torch.nn.functional as F

import fmri_models

from transforms import ThreedGaussianBlur, ThreedExposure

import wandb


def get_all_layers(net, prefix=[]):
    if hasattr(net, "_modules"):
        lst = []
        for name, layer in net._modules.items():
            full_name = "_".join((prefix + [name]))
            lst = lst + [(full_name, layer)] + get_all_layers(layer, prefix + [name])
        return lst
    else:
        return []


def save_state(net, title, output_dir):
    datestr = str(datetime.datetime.now()).replace(":", "-")
    filename = os.path.join(output_dir, f"{title}-{datestr}.pt")
    torch.save(net.state_dict(), filename)
    return filename


def get_dataset(args):
    if args.dataset.startswith("airsim"):
        split = args.dataset.split("_")
        if len(split) > 1:
            split = split[-1]
        else:
            split = "batch1"

        trainset = airsim.AirSim(
            os.path.join(args.data_root, "airsim", split),
            split="train",
            regression=not args.softmax,
        )

        tuneset = airsim.AirSim(
            os.path.join(args.data_root, "airsim", split),
            split="tune",
            regression=not args.softmax,
        )

        reportset = airsim.AirSim(
            os.path.join(args.data_root, "airsim", split),
            split="report",
            regression=not args.softmax,
        )

        train_transform = transforms.Compose(
            [
                ThreedGaussianBlur(5),
                transforms.Normalize(123.0, 75.0),
                ThreedExposure(0.3, 0.3),
            ]
        )

        eval_transform = transforms.Compose([transforms.Normalize(123.0, 75.0)])

        sz = 112
    else:
        raise NotImplementedError(f"{args.dataset} not implemented")

    return trainset, tuneset, reportset, train_transform, eval_transform, sz


def log_net(net, subnet, layers, writer, n):
    for name, layer in layers:
        if hasattr(layer, "weight"):
            writer.add_scalar(f"Weights/{name}/mean", layer.weight.mean(), n)
            writer.add_scalar(f"Weights/{name}/std", layer.weight.std(), n)
            writer.add_histogram(f"Weights/{name}/hist", layer.weight.view(-1), n)

        if hasattr(layer, "bias") and layer.bias is not None:
            writer.add_scalar(f"Biases/{name}/mean", layer.bias.mean(), n)
            writer.add_histogram(f"Biases/{name}/hist", layer.bias.view(-1), n)

    for name, param in net._parameters.items():
        writer.add_scalar(f"Weights/{name}/mean", param.mean(), n)
        writer.add_scalar(f"Weights/{name}/std", param.std(), n)
        writer.add_histogram(f"Weights/{name}/hist", param.view(-1), n)


def get_subnet(args, start_size):
    model, activations, metadata = fmri_models.get_feature_model(args)
    return model, activations, metadata


def main(args):
    print("Main")
    output_dir = os.path.join(args.output_dir, args.exp_name)
    # Train a network
    try:
        os.makedirs(args.data_root)
    except FileExistsError:
        pass

    try:
        os.makedirs(output_dir)
    except FileExistsError:
        pass

    writer = SummaryWriter(comment=args.exp_name)
    writer.add_hparams(vars(args), {})

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
    ) = get_dataset(args)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True
    )

    tuneloader = torch.utils.data.DataLoader(
        tuneset, batch_size=args.batch_size, shuffle=True, pin_memory=True
    )

    reportloader = torch.utils.data.DataLoader(
        reportset, batch_size=args.batch_size, shuffle=True, pin_memory=True
    )

    tuneloader_iter = iter(tuneloader)

    print("Init models")

    subnet, activations, metadata = get_subnet(args, start_sz)

    subnet.to(device=device)
    subnet.eval()

    # Calculate the number of features for each activation.
    X = torch.zeros(
        1, 3, 10, metadata["sz"], metadata["sz"], dtype=torch.float, device="cuda"
    )
    subnet(X)

    # Create one decoder for each layer.
    decoders = {}
    for layer_name, activation in activations.items():
        nfeats = activation.shape[1]
        if args.decoder == "average":
            net = decoder.Average(
                trainset.noutputs, trainset.nclasses, nfeats, threed=True
            ).to(device)
        elif args.decoder == "center":
            net = decoder.Center(
                trainset.noutputs, trainset.nclasses, nfeats, threed=True
            ).to(device)
        elif args.decoder == "point":
            net = decoder.Point(
                trainset.noutputs, trainset.nclasses, nfeats, threed=True
            ).to(device)
        else:
            raise NotImplementedError(f"{args.decoder} not implemented")
        decoders[layer_name] = net
        net.to(device=device)

    # Load a baseline with pre-trained weights
    if args.load_ckpt != "":
        net.load_state_dict(torch.load(args.load_ckpt))

    layers = get_all_layers(net)

    optimizers = {
        layer_name: optim.Adam(decoder.parameters(), lr=args.learning_rate)
        for layer_name, decoder in decoders.items()
    }
    scheduler = None

    net.requires_grad_(True)
    subnet.requires_grad_(True)

    if args.softmax:
        loss_fun = nn.CrossEntropyLoss()
        loss_fun_dis = nn.CrossEntropyLoss(reduction="none")
    else:
        loss_fun = nn.MSELoss()
        loss_fun_dis = nn.MSELoss(reduction="none")

    ll, m, n = 0, 0, 0
    tune_loss = {layer_name: 0.0 for layer_name in decoders.keys()}

    tuning_loss = np.zeros((len(activations), args.num_epochs, 5))
    report_loss = np.zeros((len(activations), args.num_epochs, 5))

    running_loss = 0.0
    try:
        for epoch in range(args.num_epochs):  # loop over the dataset multiple times
            for data in trainloader:
                net.train()

                # get the inputs; data is a list of [inputs, labels]
                X, labels = data
                X, labels = X.to(device), labels.to(device)
                X = fmri_models.resize(X, metadata["sz"])

                # zero the parameter gradients
                with torch.no_grad():
                    X = train_transform(X)
                    X = subnet(X)

                for k, optimizer in optimizers.items():
                    optimizer.zero_grad()
                    X = activations[k]
                    outputs = decoders[k](X)

                    loss = loss_fun(outputs, labels)

                    loss.backward()
                    optimizer.step()

                # print statistics
                running_loss += loss.item()

                if not args.softmax:
                    label_mean = labels.mean()
                    writer.add_scalar("Labels/mean", label_mean, n)

                output_mean = outputs.mean()
                writer.add_scalar("Outputs/mean", output_mean, n)

                output_std = outputs.std()
                writer.add_scalar("Outputs/std", output_std, n)

                writer.add_scalar("Loss/train", loss.item(), n)

                if ll % args.print_frequency == args.print_frequency - 1:
                    log_net(net, subnet, layers, writer, n)
                    print(
                        "[%02d, %07d] average train loss: %.3f"
                        % (epoch + 1, n, running_loss / args.print_frequency)
                    )
                    running_loss = 0
                    ll = 0

                if scheduler is not None:
                    scheduler.step()

                n += args.batch_size
                ll += 1

                if n % args.ckpt_frequency == 0:
                    save_state(net, f"model.ckpt-{n:07}", output_dir)

            def compute_test_error(loader, tl):
                tune_loss = {layer_name: 0.0 for layer_name in decoders.keys()}
                nel = 0
                for data in loader:
                    net.eval()

                    # get the inputs; data is a list of [inputs, labels]
                    with torch.no_grad():
                        X, labels = data
                        nel += X.shape[0]
                        X, labels = X.to(device), labels.to(device)

                        X = eval_transform(X)
                        X = fmri_models.resize(X, metadata["sz"])
                        X = subnet(X)

                        for k in optimizers.keys():
                            X = activations[k]
                            outputs = decoders[k](X)
                            loss = loss_fun_dis(outputs, labels).sum(0)

                            tune_loss[k] += loss.detach().cpu().numpy()

                for i, (k, optimizer) in enumerate(optimizers.items()):
                    tl[i, epoch, :] = tune_loss[k] / nel

            compute_test_error(tuneloader, tuning_loss)
            compute_test_error(reportloader, report_loss)

            total_tune_loss = tuning_loss[:, epoch, :].mean().item()
            print(f"tune accuracy: {total_tune_loss /  args.print_frequency:.3f}")

    except KeyboardInterrupt:
        pass

    best_idx = tuning_loss.mean(axis=2).argmin(axis=1)
    report_loss_ = report_loss.mean(axis=2)
    best_losses = {}
    for i, k in enumerate(optimizers.keys()):
        best_losses[k] = report_loss_[i, best_idx[i]]

    filename = save_state(net, f"model.ckpt-{n:07}", output_dir)

    if args.no_wandb:
        print("Skipping W&B per config")
    else:
        if n > 10000:
            print("Saving to weight and biases")
            wandb.init(project="crcns-train_heading.py", config=vars(args))
            config = wandb.config
            wandb.watch(net, log="all")
            wandb.log(
                {
                    "report_loss": report_loss,
                    "tuning_loss": tuning_loss,
                    "best_report_loss": best_losses,
                }
            )
            torch.save(net.state_dict(), os.path.join(wandb.run.dir, "model.pt"))
            filename = os.path.join(wandb.run.dir, "tuning_loss.npy")
            with open(filename, "wb") as f:
                np.save(f, tuning_loss)
            filename = os.path.join(wandb.run.dir, "report_loss.npy")
            with open(filename, "wb") as f:
                np.save(f, report_loss)
            print("done")
        else:
            print("Aborted too early, did not save results")


if __name__ == "__main__":
    desc = "Train a neural net"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument("--exp_name", required=True, help="Friendly name of experiment")
    parser.add_argument("--decoder", default="average", type=str, help="Decoder model")
    parser.add_argument(
        "--features",
        default="gaborpyramid3d",
        type=str,
        help="Sub-model type (currently, either xception2d, gaborpyramid2d, gaborpyramid3d",
    )
    parser.add_argument(
        "--learning_rate", default=5e-3, type=float, help="Learning rate"
    )
    parser.add_argument(
        "--num_epochs", default=20, type=int, help="Number of epochs to train"
    )
    parser.add_argument("--image_size", default=112, type=int, help="Image size")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
    parser.add_argument("--nfeats", default=64, type=int, help="Number of features")
    parser.add_argument("--num_blocks", default=0, type=int, help="Num Xception blocks")
    parser.add_argument(
        "--warmup",
        default=5000,
        type=int,
        help="Number of iterations before unlocking tuning RFs and filters",
    )
    parser.add_argument(
        "--subset",
        default="-1",
        type=str,
        help="Fit data to a specific subset of the data",
    )
    parser.add_argument(
        "--ckpt_frequency", default=2500, type=int, help="Checkpoint frequency"
    )
    parser.add_argument(
        "--print_frequency", default=100, type=int, help="Print frequency"
    )
    parser.add_argument(
        "--virtual",
        default="",
        type=str,
        help="Create virtual cells by transforming the inputs (" ", rot or all)",
    )

    parser.add_argument(
        "--no_sample",
        default=False,
        help="Whether to use a normal gaussian layer rather than a sampled one",
        action="store_true",
    )
    parser.add_argument(
        "--subsample_layers",
        default=False,
        action="store_true",
        help="Subsample layers",
    )
    parser.add_argument(
        "--no_wandb", default=False, help="Skip using W&B", action="store_true"
    )
    parser.add_argument(
        "--skip_existing", default=False, help="Skip existing runs", action="store_true"
    )
    parser.add_argument(
        "--softmax",
        default=False,
        help="Use softmax objective rather than regression",
        action="store_true",
    )

    parser.add_argument(
        "--load_conv1_weights", default="", help="Load conv1 weights in .npy format"
    )
    parser.add_argument("--load_ckpt", default="", help="Load checkpoint")
    parser.add_argument(
        "--dataset", default="airsim", help="Dataset (currently airsim only)"
    )
    parser.add_argument("--data_root", default="./data_derived", help="Data path")
    parser.add_argument("--ckpt_root", default="./checkpoints", help="Data path")
    parser.add_argument(
        "--output_dir", default="./models", help="Output path for models"
    )
    parser.add_argument(
        "--slowfast_root", default="", help="Path where SlowFast is installed"
    )
    parser.add_argument("--ntau", type=int, default=10)

    args = parser.parse_args()
    main(args)
