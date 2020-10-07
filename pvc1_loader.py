import mat_utils

import glob
import matplotlib
import matplotlib.image
import numpy as np
import os

import torch.utils.data

class PVC1(torch.utils.data.Dataset):
    """
    Loads a segment from the Ringach crcns pvc-1 dataset.

    Each call to __get_item__ returns a tuple ((X, experiment), y)

    X: a numpy array with size (3, nt, ny, nx)
    experiment: a string corresponding to which experiment we're talking about.
    y: a numpy array with size (nt - ntau + 1, nelectrodes[experiment]).

    Arguments:
        root:        the root folder where the data is stored
        nx:          the number of x values (will center crop)
        ny:          the number of y value (will center crop)
        nt:          the number of images per micro-batch
        ntau:        the number of time lags that the y response listens to
        nframedelay: the number of frames the neural response is delayed by compared to the neural data.
        split: either train or test (if test, returns a 1 / 7 test set, if train, it's the opposite)
    """
    def __init__(self, 
                 root='./crcns-ringach-data',
                 nx=224,
                 ny=224,
                 nt=20,
                 ntau=1,
                 nframedelay=2,
                 split='train',
                 ):

        if split not in ('train', 'test'):
            raise NotImplementedError('Split is set to an unknown value')

        self.nx = nx
        self.ny = ny
        self.nt = nt
        self.ntau = ntau
        self.nframedelay = nframedelay

        self.movie_info = _movie_info(root)

        paths = []
        for item in glob.glob(os.path.join(root, 'neurodata', '*', '*.mat')):
            paths.append(item)

        paths = sorted(paths)
        #paths = paths[:1]

        # Create a mapping from a single index to the necessary information needed to load the corresponding data
        sequence = []
        self.mat_files = {}
        set_num = 0

        cumulative_electrodes = 0
        for path in paths:
            mat_file = mat_utils.load_mat_as_dict(path)
            key = path
            self.mat_files[key] = mat_file

            batch = mat_file['pepANA']['listOfResults'][0]
            if batch['noRepeats'] > 1:
                n_electrodes = len(batch['repeat'][0]['data'])
            else:
                n_electrodes = len(batch['repeat']['data'])

            # Load all the conditions.
            for j, condition in enumerate(mat_file['pepANA']['listOfResults']):
                if condition['symbols'][0] != 'movie_id':
                    print(f'non-movie dataset, skipping {key}, {j}')
                    continue

                set_num += 1

                if split == 'train':
                    if set_num % 7 == 0:
                        continue
                else:
                    if set_num % 7 != 0:
                        continue

                which_movie = condition['values']
                cond = self.movie_info[tuple(which_movie)]
                nframes = cond['nframes']
                nskip = nt - ntau + 1
                ntrainings = int((nframes - nt) / nskip) + 1

                sequence += [{
                    'key': key,
                    'movie_path': os.path.join(root, 'movie_frames', f"movie{which_movie[0]:03}_{which_movie[1]:03}.images"),
                    'movie': which_movie[0],
                    'segment': which_movie[1],
                    'result': j,
                    'start_frame': nskip * i,
                    'end_frame': nskip * i + nt,
                    'electrode_range': np.arange(cumulative_electrodes, cumulative_electrodes + n_electrodes)
                } for i in range(ntrainings)]

            cumulative_electrodes += n_electrodes

        self.sequence = sequence
        self.total_electrodes = cumulative_electrodes

    def __getitem__(self, idx):
        # Load a single segment of length idx from disk.
        tgt = self.sequence[idx]

        imgs = []
        for frame in range(tgt['start_frame'], tgt['end_frame']):
            im_name = f'movie{tgt["movie"]:03}_{tgt["segment"]:03}_{frame:03}.jpeg'
            the_im = matplotlib.image.imread(os.path.join(tgt['movie_path'], im_name))
            the_im = the_im.transpose((2, 0, 1))
            assert the_im.shape[0] == 3

            cropy = (the_im.shape[1] - self.ny) // 2
            cropx = (the_im.shape[2] - self.nx) // 2
            rgy = slice(cropy, the_im.shape[1] - cropy)
            rgx = slice(cropx, the_im.shape[2] - cropx)

            imgs.append(the_im[:, rgy, rgx])

        X = np.stack(imgs, axis=-1).astype(np.float32)
        mat_file = self.mat_files[tgt['key']]
        
        y = []
        bins = np.arange(tgt['start_frame'] + self.nframedelay + self.ntau, 
                         tgt['start_frame'] + self.nframedelay + self.nt + 2) / 30.0
        batch = mat_file['pepANA']['listOfResults'][tgt['result']]

        if batch['noRepeats'] > 1:
            n_electrodes = len(batch['repeat'][0]['data'])
            for el in range(n_electrodes):
                y_ = 0
                for i in range(len(batch['repeat'])):
                    # Bin the total number of spikes. This is simply the multi-unit activity.
                    d_, _ = np.histogram(batch['repeat'][i]['data'][el][0], bins)
                    y_ += d_
                y.append(y_ / float(len(batch['repeat'])))
        else:
            n_electrodes = len(batch['repeat']['data'])
            for el in range(n_electrodes):
                # Bin the total number of spikes. This is simply the multi-unit activity.
                d_, _ = np.histogram(batch['repeat']['data'][el][0], bins)
                y.append(d_)

        y = np.array(y).T.astype(np.float32)

        assert X.shape[-1] == self.nt
        assert y.shape[0] == self.nt - self.ntau + 1
        assert X.shape[0] == 3
        assert X.shape[1] == self.ny
        assert X.shape[2] == self.nx
        return ((X, tgt['electrode_range']), y)

    def __len__(self):
        # Returns the length of a dataset
        return len(self.sequence)

def _movie_info(root):
    """
    Build up a hashmap from tuples of (movie, segment) to info about the 
    movie, including location and duration
    """
    movie_info = {}
    for i in range(30):
        for j in range(4):
            root_ = os.path.join(root, "movie_frames", f"movie{j:03}_{i:03}.images")
            with open(os.path.join(root_, 'nframes'), 'r') as f:
                nframes = int(f.read())
            
            movie_info[(j, i)] = {'nframes': nframes,
                                  'root': root}
    
    return movie_info

if __name__ == '__main__':
    dataset_train = PVC1(split='train')
    train_electrodes = dataset_train.total_electrodes
    train_len = len(dataset_train)
    print("Len(train_dataset): ", train_len)
    for i in range(0, len(dataset_train), 100):
        d = dataset_train[i]
        assert len(d) == 2
        assert len(d[0]) == 2
        assert d[0][1].ndim == 1
        assert d[0][1].size >= 1
        assert d[1].shape[0] == d[0][0].shape[-1]

    dataset_test = PVC1(split='test')
    test_electrodes = dataset_test.total_electrodes
    test_len = len(dataset_test)
    assert train_electrodes == test_electrodes

    print("Len(test_dataset): ", test_len)
    for i in range(0, len(dataset_test), 100):
        d = dataset_test[i]

    assert abs((train_len + test_len) / test_len - 7) < .1