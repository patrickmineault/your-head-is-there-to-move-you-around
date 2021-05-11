from pathlib import Path

import numpy as np
from skimage.draw import disk
from skimage.io import imsave
from PIL import Image

from tqdm import tqdm

import tables
from skimage.transform import rescale, resize, downscale_local_mean

import sys

sys.path.append("../")
from derive_dataset import get_max_r2


def generate_hyperflow_sequence_fast(f):
    # Nominal framerate
    fps = 30
    sz = 448
    ds = 4  # Downsample by what factor.

    # The paper says .1 degrees diameter. However, I measured .4 degrees in a video from
    # that era.
    # Radius.
    lifetime = 3  # In frames.

    stimidx = f.get_node("/stimidx_hf")[:] - 1
    gridx = f.get_node("/gridx_hf")[:]
    Y = f.get_node("/Y_hf")[:]
    dx = gridx.max() - gridx.min()
    # It's unclear from the documentation whether is was the number of dots or
    # the dot size which was changed when the stimulus was made smaller. Assume
    # it's a smaller dot.

    ndots = 200
    disk_sz_deg = 0.4 / 2 * (dx / 48)  # in degrees
    disk_sz = int(sz * disk_sz_deg / dx + 1)  # round up
    assert disk_sz > 0

    fields = f.get_node("/stim_hf")[:].squeeze()
    field_side = int(np.sqrt(fields.shape[0] // 2).item())

    ims = []
    for frame in range(fields.shape[-1]):
        stim = (
            fields[:, frame].reshape((2, field_side, field_side)).transpose((2, 1, 0))
        )

        # Upscale.
        xv = np.array(Image.fromarray(stim[:, :, 0]).resize((sz, sz), Image.BILINEAR))
        yv = -np.array(Image.fromarray(stim[:, :, 1]).resize((sz, sz), Image.BILINEAR))

        mask = np.array(
            Image.fromarray(
                (255 * (abs(stim[:, :, 0]) + abs(stim[:, :, 1]) > 0).astype(np.uint8))
            ).resize((sz, sz), Image.BILINEAR)
        )

        # Apply a mask. Note the inherent mask smoothing.
        mask = mask > 64

        # Apply a mask.
        vidx, hidx = np.nonzero(mask)

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
        rr, cc = disk((img.shape[0] // 2, img.shape[1] // 2), disk_sz, shape=img.shape)
        rr = rr - img.shape[0] // 2
        cc = cc - img.shape[1] // 2

        dotr = (star_pos[:, 1].reshape((-1, 1)) + rr.reshape((1, -1))).ravel()
        dotc = (star_pos[:, 0].reshape((-1, 1)) + cc.reshape((1, -1))).ravel()

        validx = (
            (dotr >= 0) & (dotr < img.shape[0]) & (dotc >= 0) & (dotc < img.shape[1])
        )

        dotr = dotr[validx].astype(np.int)
        dotc = dotc[validx].astype(np.int)

        img = np.zeros((sz, sz), dtype=np.uint8)
        img[dotr, dotc] = 255
        img = img * mask

        # ds-fold antialiasing.
        img = (
            img.reshape((sz // ds, ds, sz // ds, ds))
            .mean(axis=3)
            .mean(axis=1)
            .astype(np.uint8)
        )
        ims.append(img)
        # imsave(f"figures/hf/seq_{(frame):05}.png", img)

    M = np.stack(ims, axis=0)
    # M = np.stack([M, M, M], axis=1)
    assert M.shape == (fields.shape[-1], sz // ds, sz // ds)

    return (M, stimidx.T, Y)


def generate_unmatched_sequence(f, stem):
    X_hf, Xidx_hf, Y_hf = generate_hyperflow_sequence_fast(f)
    Y_hf = Y_hf.T

    t = f.get_node("/t")[:].ravel()
    assert Xidx_hf.shape[0] == Y_hf.shape[0]

    # Designate a certain proportion of the sequence as the train set and tune
    # set.

    block_len = 6  # in seconds
    nfolds = 10

    reportidx = (np.floor(t / block_len) % nfolds) == 4

    print(f"In report fold: {reportidx.sum()}/{reportidx.size}")

    fout = tables.open_file(f"/mnt/e/data_derived/crcns-mt1/movies/{stem}.h5", "w")
    fout.create_array("/", "X_traintune", obj=X_hf)
    fout.create_array("/", "Xidx_traintune", obj=Xidx_hf[~reportidx, :])
    fout.create_array("/", "Y_traintune", obj=Y_hf[~reportidx, :])

    Xidx_report = Xidx_hf[reportidx, :]
    Y_report = Y_hf[reportidx, :]

    # fout.create_array("/", "X_report", obj=X_hf)
    fout.create_array("/", "Xidx_report", obj=Xidx_report)
    fout.create_array("/", "Y_report", obj=Y_report)

    fout.close()


def generate_unmatched_sequences():
    files = Path("/mnt/e/data_derived/crcns-mt1/designmats/").glob("*.mat")
    files = sorted(files)

    i = 0
    for filename in tqdm(files):
        i += 1
        if i >= 48:
            continue
        f = tables.open_file(filename)
        generate_unmatched_sequence(f, filename.stem)
        f.close()


if __name__ == "__main__":
    # generate_matched_sequences()
    # get_all_maxr2()
    # generate_long_sequence()
    generate_unmatched_sequences()
