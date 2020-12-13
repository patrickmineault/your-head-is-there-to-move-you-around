import mat_utils

import datetime
import glob
import matplotlib
import matplotlib.image
import numpy as np
import os
import requests
import subprocess
import time

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
        nframestart: the number of frames after the onset of a sequence to start at. 15 by default ~ 500ms
        split: either train, tune or test (if tune or test, returns a 1 / 10 tune/test set, if train, it's the opposite)
    """
    def __init__(self, 
                 root='./crcns-ringach-data',
                 nx=224,
                 ny=224,
                 nt=20,
                 ntau=1,
                 nframedelay=2,
                 nframestart=15,
                 split='train',
                 ):

        if split not in ('train', 'tune', 'test'):
            raise NotImplementedError('Split is set to an unknown value')

        if ntau + nframedelay > nframestart:
            raise NotImplementedError('ntau + nframedelay > nframestart, sequence starts before frame 0')

        self.nx = nx
        self.ny = ny
        self.nt = nt
        self.ntau = ntau
        self.nframedelay = nframedelay
        self.nframestart = nframestart

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

        splits = {'train': [0, 1, 2, 3, 5, 6, 7, 8],
                  'test': [9],
                  'tune': [4]}

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

                # Make sure that at most 10 consecutive seconds are in each 
                # split to prevent leakage.
                if int(set_num / 10) % 10 not in splits[split]:
                    continue

                which_movie = condition['values']
                cond = self.movie_info[tuple(which_movie)]
                nframes = cond['nframes']
                nskip = nt
                ntrainings = int(np.ceil((nframes - nt) / nskip) + 1)

                for start_time in range(self.nframestart, nframes, nskip):
                    if start_time + nskip + 1 > nframes:
                        # Incomplete frame.
                        continue

                    end_time = min((nframes, start_time + nskip + 1))
                    spike_frames = np.arange(start_time, 
                                             end_time)
                    bins = spike_frames / 30.0
                    sequence.append({
                        'key': key,
                        'movie_path': os.path.join(root, 
                                                   'movie_frames', 
                                                   f"movie{which_movie[0]:03}_{which_movie[1]:03}.images"),
                        'movie': which_movie[0],
                        'segment': which_movie[1],
                        'result': j,
                        'start_frame': start_time - self.nframedelay - self.ntau + 2,
                        'end_frame': end_time - self.nframedelay,
                        'electrode_range': np.arange(cumulative_electrodes, cumulative_electrodes + n_electrodes),
                        'bins': bins,
                        'spike_frames': spike_frames,
                        'nframes': nframes})

            cumulative_electrodes += n_electrodes

        self.sequence = sequence
        self.total_electrodes = cumulative_electrodes

    def __getitem__(self, idx):
        # Load a single segment of length idx from disk.
        tgt = self.sequence[idx]
        bins = tgt['bins']

        imgs = []
        # The images are natively 320 x 240.
        for frame in range(tgt['start_frame'], tgt['end_frame']):
            if frame < 0 or frame >= tgt['nframes']:
                imgs.append(128 * np.ones((3, self.ny, self.nx)))
                continue

            im_name = f'movie{tgt["movie"]:03}_{tgt["segment"]:03}_{frame:03}.jpeg'
            the_im = matplotlib.image.imread(os.path.join(tgt['movie_path'], im_name))
            the_im = the_im.transpose((2, 0, 1))
            assert the_im.shape[0] == 3

            cropy = (the_im.shape[1] - self.ny) // 2
            cropx = (the_im.shape[2] - self.nx) // 2
            rgy = slice(the_im.shape[1] - self.ny, the_im.shape[1])
            
            # All the receptive fields are on the right hand side
            rgx = slice(cropx * 2, the_im.shape[2])

            imgs.append(the_im[:, rgy, rgx])

        # Center and normalize.
        # This seems like a random order, but it's to fit with the ordering
        # the standard ordering of conv3d. 
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html
        X = (np.stack(imgs, axis=1).astype(np.float32) - 128.0) / 128.0
        mat_file = self.mat_files[tgt['key']]
        
        y = []
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

        # Create a mask from the electrode range
        m = np.zeros((self.total_electrodes), dtype=np.bool)
        m[tgt['electrode_range']] = True

        Y = np.zeros((self.total_electrodes, y.shape[0]))
        Y[tgt['electrode_range'], :] = y.T

        return (X, m, Y)

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

def download(root, url=None):
    """Download the dataset to disk.
    
    Arguments:
        root: root folder to download to.
        url: the root URL to grab the data from.

    Returns:
        True if downloaded correctly
    """
    if url is None:
        url = os.getenv('GCS_ROOT')

    zip_name = 'crcns-pvc1.zip'

    out_file = os.path.join(root, 'zip', zip_name)
    if os.path.exists(out_file) and os.stat(out_file).st_size == 1798039870:
        print(f"Already fetched {zip_name}")
    else:
        try:
            os.makedirs(os.path.join(root, 'zip'))
        except FileExistsError:
            pass

        # Instead of downloading in Python and taking up a bunch of memory, use curl.
        process = subprocess.Popen(['wget', 
                                    '--quiet',
                                    url + zip_name,
                                    '-O',
                                    out_file], 
                                    stdout=subprocess.DEVNULL)

        t0 = datetime.datetime.now()
        progress = '|\\-/'
        while process.poll() is None:
            dt = (datetime.datetime.now() - t0) / datetime.timedelta(seconds=.5)
            char = progress[int(dt) % 4]
            print('\r' + char, end='')
            time.sleep(.5)
        print('\n')

        # Check everything good
        if not os.path.exists(out_file):
            # Something bad happened during download
            print(f"Failed to download {zip_name}")
            return False

    # Now unzip the data if necessary.
    if os.path.exists(os.path.join(root, 'crcns-ringach-data')):
        print("Already unzipped")
        return True
    else:
        process = subprocess.Popen(['unzip', 
                                    out_file,
                                    '-d',
                                    root],
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE)

        process.communicate()
        return True
    

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