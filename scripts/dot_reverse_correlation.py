from pathlib import Path

import napari
import numpy as np
from skimage.draw import disk
from skimage.io import imsave
from PIL import Image

from tqdm import tqdm

import tables
from skimage.transform import rescale, resize, downscale_local_mean

import sys
import scipy
import scipy.ndimage

sys.path.append("../")
from fmri_models import get_feature_model

import collections
import torch

from python_dict_wrapper import wrap
import pickle


def generate_dot_sequences(D):
    # Nominal framerate
    fps = 30
    sz = 448
    ds = 4  # Downsample by what factor.
    nt = 16

    # The paper says .1 degrees diameter. However, I measured .4 degrees in a video from
    # that era.
    # Radius.
    lifetime = 5  # In frames.

    gridx, gridy = np.meshgrid(np.arange(sz), np.arange(sz))
    # It's unclear from the documentation whether is was the number of dots or
    # the dot size which was changed when the stimulus was made smaller. Assume
    # it's a smaller dot.

    hidx = gridx.ravel()
    vidx = gridy.ravel()

    gridx = (gridx - gridx.mean()) / 2.5
    gridy = (gridy - gridy.mean()) / 2.5

    dx = gridx.max() - gridx.min()
    ndots = 400
    disk_sz_deg = 0.8 / 2 * (dx / 48)  # in degrees
    disk_sz = int(sz * disk_sz_deg / dx + 1)  # round up
    assert disk_sz > 0

    I = []
    for i in range(D.shape[0]):
        xv = D[i, :, :, 0]
        yv = D[i, :, :, 1]
        ims = []
        for frame in range(nt):
            if frame == 0:
                # Sample positions with replacement.
                if len(vidx) > 0:
                    dot_idx = np.random.randint(0, len(vidx), ndots)

                    assert len(dot_idx) == ndots

                    star_pos = np.zeros((ndots, 2), dtype=np.float)
                    star_pos[:, 0] = hidx[dot_idx]
                    star_pos[:, 1] = vidx[dot_idx]
                else:
                    # Occasionally, the first frame may be empty.
                    star_pos = np.zeros((ndots, 2), dtype=np.float)
            else:
                # Replace expired dots.
                replace_idx = (
                    np.arange(
                        frame * (ndots // lifetime), (frame + 1) * (ndots // lifetime)
                    )
                    % ndots
                )

                if len(vidx) > 0:
                    # The stimulus can be wholly off-screen.
                    dot_idx = np.random.randint(0, len(vidx), len(replace_idx))
                    assert len(dot_idx) == len(replace_idx)

                    star_pos[replace_idx, 0] = hidx[dot_idx]
                    star_pos[replace_idx, 1] = vidx[dot_idx]

            # Advance the stars.
            speed_mult = (xv.shape[0] / dx) / fps  # pixels / degrees / frame * s

            # xv is in degrees / s
            # and star_pos is in pixels
            # hence speed_mult is in s / degrees * pixels

            ypos = star_pos[:, 1].copy()
            xpos = star_pos[:, 0].copy()

            star_pos[:, 0] += (
                speed_mult * xv[ypos.astype(np.int) % sz, xpos.astype(np.int) % sz]
            )
            star_pos[:, 1] += (
                speed_mult * yv[ypos.astype(np.int) % sz, xpos.astype(np.int) % sz]
            )

            # Now render the field.
            img = np.zeros((sz, sz), dtype=np.uint8)
            rr, cc = disk(
                (img.shape[0] // 2, img.shape[1] // 2), disk_sz, shape=img.shape
            )
            rr = rr - img.shape[0] // 2
            cc = cc - img.shape[1] // 2

            dotr = (star_pos[:, 1].reshape((-1, 1)) + rr.reshape((1, -1))).ravel()
            dotc = (star_pos[:, 0].reshape((-1, 1)) + cc.reshape((1, -1))).ravel()

            validx = (
                (dotr >= 0)
                & (dotr < img.shape[0])
                & (dotc >= 0)
                & (dotc < img.shape[1])
            )

            dotr = dotr[validx].astype(np.int)
            dotc = dotc[validx].astype(np.int)

            img = np.zeros((sz, sz), dtype=np.uint8)
            img[dotr, dotc] = 255

            # ds-fold antialiasing.
            img = (
                img.reshape((sz // ds, ds, sz // ds, ds))
                .mean(axis=3)
                .mean(axis=1)
                .astype(np.uint8)
            )
            ims.append(img)
        M = np.stack(ims, axis=0)
        I.append(M)

    seq = np.stack(I, axis=0)

    return np.stack([seq, seq, seq], axis=1)


def forward_each(model, hooks, X, D):
    model.eval()
    with torch.no_grad():
        _ = model(torch.tensor(X, dtype=torch.float))

    responses = {}
    for key in hooks.keys():
        l = hooks[key]
        l = l[:, :, l.shape[2] // 2, l.shape[3] // 2, l.shape[4] // 2]
        l = l.cpu().detach().numpy()
        latents = (l - l.mean(axis=0, keepdims=True)).T @ D.reshape((D.shape[0], -1))
        latents = latents.reshape((l.shape[1],) + D.shape[1:])
        responses[key] = latents

    return responses


def combine(r0, r1):
    responses = {}
    for key in r0.keys():
        responses[key] = r0[key] + r1[key]

    return responses


def speed_tune(D):
    D = D.copy()
    speed = np.log(1 + np.sqrt((D ** 2).sum(axis=3)))
    angle = np.arctan2(D[:, :, :, 1], D[:, :, :, 0])
    D[:, :, :, 0] = speed * np.cos(angle)
    D[:, :, :, 1] = speed * np.sin(angle)
    return D


if __name__ == "__main__":
    args = wrap(
        {
            "features": "airsim_04",
            "ckpt_root": "../pretrained",
            "slowfast_root": "../../slowfast",
            "ntau": 16,
            "nt": 1,
            "subsample_layers": False,
        }
    )

    model, hooks, data = get_feature_model(args)

    for j in tqdm(range(3600 * 4)):
        sz = 448
        D = np.random.randn(32, 28, 28, 2) * 50
        DD = np.zeros((D.shape[0], sz, sz, 2))
        for i in range(D.shape[0]):

            xv = np.array(
                Image.fromarray(D[i, :, :, 0]).resize((sz, sz), Image.BILINEAR)
            )
            yv = -np.array(
                Image.fromarray(D[i, :, :, 1]).resize((sz, sz), Image.BILINEAR)
            )

            DD[i, :, :, 0] = xv
            DD[i, :, :, 1] = yv

        images = generate_dot_sequences(DD)
        responses = forward_each(model, hooks, images / 100.0, speed_tune(D))
        if j == 0:
            all_responses = responses
        else:
            all_responses = combine(responses, all_responses)

    with open("rc_big_speedtuned.pkl", "wb") as f:
        pickle.dump(all_responses, f)
