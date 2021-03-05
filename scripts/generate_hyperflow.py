from pathlib import Path

import numpy as np
from skimage.draw import disk
from skimage.io import imsave
from PIL import Image

from tqdm import tqdm

import tables
from skimage.transform import rescale, resize, downscale_local_mean


def generate_supertune_sequence(f):
    # Nominal framerate
    fps = 30
    sz = 448
    nframes = 10
    ds = 4  # Downsample by what factor.

    # The paper says .1 degrees diameter. However, I measured .4 degrees in a video from
    # that era.
    # Radius.
    ndots = 400
    lifetime = 6  # In frames.

    gridx = f.get_node("/gridx_st")[:]
    dx = gridx.max() - gridx.min()
    disk_sz_deg = 0.4 / 2 * (dx / 48)  # in degrees

    disk_sz = int(sz * disk_sz_deg / dx + 1)  # round up
    assert disk_sz > 0

    Y_st = f.get_node("/Y_st")[:].T
    Yall_st = f.get_node("/Yall_st")[:].T
    fields = f.get_node("/X_st")[:]

    assert Y_st.shape[0] == 216

    field_side = int(np.sqrt(fields.shape[0] // 2).item())

    Ms = []
    for f in range(fields.shape[1]):
        stim = fields[:, f].reshape((2, field_side, field_side)).transpose((2, 1, 0))

        # Upscale.

        xv = np.array(Image.fromarray(stim[:, :, 0]).resize((sz, sz), Image.NEAREST))
        yv = -np.array(Image.fromarray(stim[:, :, 1]).resize((sz, sz), Image.NEAREST))

        mask = abs(xv) + abs(yv) > 0

        xrg = np.where((abs(xv) + abs(yv)).sum(axis=0))[0]
        yrg = np.where((abs(xv) + abs(yv)).sum(axis=1))[0]

        # print(xrg.min(), yrg.min())

        star_pos = np.random.uniform(size=(ndots, 2))
        star_pos[:, 0] = xrg.min() + (xrg.max() - xrg.min()) * star_pos[:, 0]
        star_pos[:, 1] = yrg.min() + (yrg.max() - yrg.min()) * star_pos[:, 1]

        ims = []
        for i in range(nframes):
            speed_mult = (xv.shape[0] / dx) / fps  # pixels / degrees / frame * s

            # print(star_pos[:, 0].astype(np.int))

            ypos = star_pos[:, 1].copy()
            xpos = star_pos[:, 0].copy()
            star_pos[:, 0] += (
                speed_mult * xv[ypos.astype(np.int) % sz, xpos.astype(np.int) % sz]
            )
            star_pos[:, 1] += (
                speed_mult * yv[ypos.astype(np.int) % sz, xpos.astype(np.int) % sz]
            )

            img = np.zeros((sz, sz), dtype=np.uint8)
            for x_, y_ in zip(star_pos[:, 0], star_pos[:, 1]):
                if x_ < 0 or y_ < 0 or x_ >= sz or y_ >= sz:
                    continue

                if mask[int(y_), int(x_)]:
                    rr, cc = disk((y_, x_), disk_sz, shape=img.shape)
                    img[rr, cc] = 255

            img = (
                img.reshape((sz // ds, ds, sz // ds, ds))
                .mean(axis=3)
                .mean(axis=1)
                .astype(np.uint8)
            )
            ims.append(img)
            # imsave(f"figures/seq_{f:03}_{(i):02}.png", img)

            star_pos_ = np.random.uniform(size=(ndots // lifetime, 2))

            idx = (
                np.arange(i * (ndots // lifetime), (i + 1) * (ndots // lifetime))
                % ndots
            )

            star_pos[idx, 0] = xrg.min() + (xrg.max() - xrg.min()) * star_pos_[:, 0]
            star_pos[idx, 1] = yrg.min() + (yrg.max() - yrg.min()) * star_pos_[:, 1]

        M = np.stack(ims, axis=0)
        Ms.append(M)

    M = np.concatenate(Ms, axis=0)

    Xidx = nframes * np.arange(216).reshape((-1, 1)) + np.arange(nframes).reshape(
        (1, -1)
    )
    assert len(np.unique(Xidx)) == Xidx.size
    assert Xidx.size == M.shape[0]

    return M, Xidx, Y_st, Yall_st


def generate_hyperflow_sequence(f):
    # Nominal framerate
    fps = 30
    sz = 448
    ds = 4  # Downsample by what factor.

    # The paper says .1 degrees diameter. However, I measured .4 degrees in a video from
    # that era.
    # Radius.
    lifetime = 6  # In frames.

    stimidx = f.get_node("/stimidx_hf")[:] - 1
    gridx = f.get_node("/gridx_hf")[:]
    Y = f.get_node("/Y_hf")[:]
    dx = gridx.max() - gridx.min()
    # It's unclear from the documentation whether is was the number of dots or
    # the dot size which was changed when the stimulus was made smaller. Assume
    # it's a smaller dot.
    ndots = 400
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

        # Apply a mask.
        mask = (abs(xv) + abs(yv)) > 0
        vidx, hidx = np.nonzero(mask)

        if frame == 0:
            # Sample positions with replacement.
            dot_idx = np.random.randint(0, len(vidx), ndots)
            assert len(dot_idx) == ndots

            star_pos = np.zeros((ndots, 2), dtype=np.float)
            star_pos[:, 0] = hidx[dot_idx]
            star_pos[:, 1] = vidx[dot_idx]
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
        for x_, y_ in zip(star_pos[:, 0], star_pos[:, 1]):
            if x_ < 0 or y_ < 0 or x_ >= sz or y_ >= sz:
                continue

            if mask[int(y_), int(x_)]:
                rr, cc = disk((y_, x_), disk_sz, shape=img.shape)
                img[rr, cc] = 255

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


def generate_matched_sequence(f, stem):
    X_st, Xidx_st, Y_st, Yall_st = generate_supertune_sequence(f)
    X_hf, Xidx_hf, Y_hf = generate_hyperflow_sequence(f)

    assert Xidx_st.shape[1] == Xidx_hf.shape[1]
    assert Xidx_st[0, 1] > Xidx_st[0, 0]
    assert Xidx_hf[0, 1] > Xidx_hf[0, 0]

    signal_power = (
        1
        / (Yall_st.shape[0] - 1)
        * (Yall_st.shape[0] * Yall_st.mean(0).var() - Yall_st.var(1).mean())
    )
    response_power = Yall_st.mean(0).var()
    corr_multiplier = np.sqrt(response_power / signal_power)

    fout = tables.open_file(f"/mnt/e/data_derived/packlab-mst/{stem}.h5", "w")
    fout.create_array("/", "X_traintune", obj=X_hf)
    fout.create_array("/", "Xidx_traintune", obj=Xidx_hf)
    fout.create_array("/", "Y_traintune", obj=Y_hf)

    fout.create_array("/", "X_report", obj=X_st)
    fout.create_array("/", "Xidx_report", obj=Xidx_st)
    fout.create_array("/", "Y_report", obj=Y_st)
    fout.create_array("/", "Yall_report", obj=Y_st)
    fout.create_array("/", "corr_multiplier", obj=corr_multiplier)
    fout.close()


def generate_matched_sequences():
    files = Path("/mnt/e/data_derived/packlab-mst/").glob("*.mat")
    files = sorted(files)

    for filename in tqdm(files):
        f = tables.open_file(filename)
        if f.get_node("/stmatcheshf")[:]:
            generate_matched_sequence(f, filename.stem)
        f.close()


if __name__ == "__main__":
    generate_matched_sequences()