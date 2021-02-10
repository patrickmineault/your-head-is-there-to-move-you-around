from pathlib import Path

import numpy as np
from skimage.draw import disk
from skimage.io import imsave
from PIL import Image

import tables
from skimage.transform import rescale, resize, downscale_local_mean


def generate_starfield_sequences():
    # Nominal framerate
    fps = 30

    # Nominal velocity
    mulvel = 1.5  # Multiple velocity
    nframes = 12
    disk_sz = 4  # in pixels
    ndots = 150
    sz = 224 * 2
    lifetime = 6

    files = Path("/mnt/e/data_derived/packlab-mst/").glob("*.mat")
    files = sorted(files)
    Y = []
    for filename in files:
        f = tables.open_file(filename)

        try:
            fields = f.get_node("/Y_st")[:].squeeze()
            Y.append(fields)
        except tables.exceptions.NoSuchNodeError:
            pass

        f.close()

    Y = np.array(Y).T
    assert Y.shape[0] == 216

    f = tables.open_file("/mnt/e/data_derived/packlab-mst/ju334.mat")
    fields = f.get_node("/X_st")[:]
    f.close()

    Ms = []
    for f in range(fields.shape[1]):
        stim = fields[:, f].reshape((2, 24, 24)).transpose((2, 1, 0))

        # Upscale.

        xv = np.array(Image.fromarray(stim[:, :, 0]).resize((sz, sz), Image.NEAREST))
        yv = -np.array(Image.fromarray(stim[:, :, 1]).resize((sz, sz), Image.NEAREST))

        xrg = np.where((abs(xv) + abs(yv)).sum(axis=0))[0]
        yrg = np.where((abs(xv) + abs(yv)).sum(axis=1))[0]

        print(xrg.min(), yrg.min())

        star_pos = np.random.uniform(size=(ndots, 2))
        star_pos[:, 0] = xrg.min() + (xrg.max() - xrg.min()) * star_pos[:, 0]
        star_pos[:, 1] = yrg.min() + (yrg.max() - yrg.min()) * star_pos[:, 1]
        star_pos = star_pos.astype(np.int)

        ims = []
        for i in range(nframes):
            vel = (i - (nframes - 1) / 2) / fps * mulvel
            # print(star_pos[:, 0].astype(np.int))

            x = vel * xv[star_pos[:, 1], star_pos[:, 0]] + star_pos[:, 0]
            y = vel * yv[star_pos[:, 1], star_pos[:, 0]] + star_pos[:, 1]

            img = np.zeros((sz, sz), dtype=np.uint8)
            for x_, y_ in zip(x, y):
                rr, cc = disk((y_, x_), disk_sz, shape=img.shape)
                img[rr, cc] = 255

            img = (
                img.reshape((112, 4, 112, 4)).mean(axis=3).mean(axis=1).astype(np.uint8)
            )
            ims.append(img)
            imsave(f"figures/seq_{f:03}_{(i):02}.png", img)

            star_pos_ = np.random.uniform(size=(ndots // lifetime, 2))

            idx = (
                np.arange(i * (ndots // lifetime), (i + 1) * (ndots // lifetime))
                % ndots
            )

            star_pos[idx, 0] = xrg.min() + (xrg.max() - xrg.min()) * star_pos_[:, 0]
            star_pos[idx, 1] = yrg.min() + (yrg.max() - yrg.min()) * star_pos_[:, 1]

        M = np.stack(ims, axis=0)
        M = np.stack([M, M, M], axis=0)
        Ms.append(M)

    M = np.stack(Ms, axis=0)

    f = tables.open_file("/mnt/e/data_derived/packlab-st/starfields.h5", "w")
    f.create_array("/", "stim", obj=M)
    # f.create_array('/', 'labels', obj=np.array(headings))
    f.close()

    #


def generate_responses():
    files = Path("/mnt/e/data_derived/packlab-mst/").glob("*.mat")
    files = sorted(files)
    Y = []
    for filename in files:
        f = tables.open_file(filename)

        try:
            fields = f.get_node("/Y_st")[:].squeeze()
            Y.append(fields)
        except tables.exceptions.NoSuchNodeError:
            pass

        f.close()

    Y = np.array(Y).T
    assert Y.shape[0] == 216
    f = tables.open_file("/mnt/e/data_derived/packlab-st/MSTd_resp.h5", "w")
    f.create_array("/", "resp", obj=Y)
    f.close()

    # f = tables.open_file("/mnt/e/data_derived/packlab-st/V3A.h5", "w")
    # f.create_array("/", "resp", obj=Y)
    # f.close()


if __name__ == "__main__":
    # generate_starfield_sequences()

    generate_responses()