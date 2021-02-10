import numpy as np
from skimage.draw import disk
from skimage.io import imsave
import tables

def generate_starfield_sequences():
    # Nominal framerate
    fps = 30

    # Nominal velocity
    vel = 1  # meters per second
    nframes = 12
    dot_size = .006 # in meters
    clip_plane = .5
    ndots = 900
    sz = 224*2

    star_pos = np.random.uniform(size=(ndots, 3))
    star_pos[:, :2] = 2 * (star_pos[:, :2] - .5)
    star_pos[:, -1] = star_pos[:, -1] * .5 + clip_plane

    # Forward heading direction
    headings = [-180, -135, -90, -45, -22.5, 0, 22.5, 45, 90, 135]
    Ms = []
    for heading in headings:
        ims = []
        for i in range(nframes):
            p = (i - nframes/2) / nframes
            zp  = vel * i / (nframes - 1) / fps * np.cos(heading / 180 * np.pi)
            xp  = vel * i / (nframes - 1) / fps * np.sin(heading / 180 * np.pi)
            val_p = star_pos - np.array([xp, 0, zp]).reshape((1, -1))
            #val_p = p[p[:, 2] > clip_plane]
            x = val_p[:, 0] / val_p[:, 2]
            y = val_p[:, 1] / val_p[:, 2]
            x = (x + 1) / 2 * sz  # This gives a field of view of 90 degrees
            y = (y + 1) / 2 * sz
            disk_sz = dot_size / val_p[:, 2] * sz
            
            img = np.zeros((sz, sz), dtype=np.uint8)
            for x_, y_, disk_sz_ in zip(x, y, disk_sz):
                rr, cc = disk((y_, x_), disk_sz_, shape=img.shape)
                img[rr, cc] = 255

            img = img.reshape((112, 4, 112, 4)).mean(axis=3).mean(axis=1).astype(np.uint8)
            ims.append(img)

        M = np.stack(ims, axis=0)
        M = np.stack([M, M, M], axis=0)
        Ms.append(M)

    M = np.stack(Ms, axis=0)

    f = tables.open_file('/mnt/e/data_derived/crcns-stc1/stc1-starfields.h5', 'w')
    f.create_array('/', 'stim', obj=M)
    f.create_array('/', 'labels', obj=np.array(headings))
    f.close()

    #imsave(f'figures/seq_{heading}_{(i + nframes//2):02}.png', img)

if __name__ == '__main__':
    generate_starfield_sequences()