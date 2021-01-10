from . import utils

import datetime
import glob
import matplotlib
import matplotlib.image
import numpy as np
import os
import requests
import subprocess
import tables
import time

import torch.utils.data

movie_cache = {}

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
        split: either train, tune or report (if tune or report, returns a 1 / 10 tune/report set, if train, 8/10)
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

        framerate = 30.0

        if split not in ('train', 'tune', 'report'):
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
        self.root = root

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
                  'tune': [4],
                  'report': [9],
                  }

        nblocks = 10

        cumulative_electrodes = 0
        nrepeats = []
        for path in paths:
            mat_file = utils.load_mat_as_dict(path)
            key = path
            self.mat_files[key] = mat_file

            batch = mat_file['pepANA']['listOfResults'][0]
            if batch['noRepeats'] > 1:
                n_electrodes = len(batch['repeat'][0]['data'])
            else:
                n_electrodes = len(batch['repeat']['data'])

            # Load all the conditions.
            n_electrodes_seen = 0
            for j, condition in enumerate(mat_file['pepANA']['listOfResults']):
                if condition['symbols'][0] != 'movie_id':
                    # This is not movie data, skip.
                    #print(f'non-movie dataset, skipping {key}, {j}')
                    continue

                if n_electrodes_seen == 0:
                    nrepeats += [batch['noRepeats']] * n_electrodes

                n_electrodes_seen = n_electrodes

                set_num += 1

                # The train, tune and report splits
                if set_num % nblocks not in splits[split]:
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
                    bins = spike_frames / framerate
                    for i in range(n_electrodes):
                        # Although this data was recorded multiple electrodes at a time, give it one electrode at a time
                        # to fit better with other data, e.g. Jack's
                        sequence.append({
                            'key': key,
                            'movie_path': os.path.join(root, 
                                                    'movie_frames', 
                                                    f"movie{which_movie[0]:03}_{which_movie[1]:03}.images"),
                            'movie': which_movie[0],
                            'segment': which_movie[1],
                            'result': j,
                            'start_frame': start_time - self.nframedelay - self.ntau,
                            'end_frame': end_time - self.nframedelay - 2,
                            'abs_electrode_num': cumulative_electrodes + i,
                            'rel_electrode_num': i,
                            'bins': bins,
                            'spike_frames': spike_frames,
                            'nframes': nframes})

            cumulative_electrodes += n_electrodes_seen

        self.nrepeats = np.array(nrepeats)
        self.sequence = sequence
        self.total_electrodes = cumulative_electrodes

    def __getitem__(self, idx):
        # Load a single segment of length idx from disk.
        tgt = self.sequence[idx]
        bins = tgt['bins']

        global movie_cache

        # Lazy load the set of images.
        index = (tgt['movie'], tgt['segment'])
        if index not in movie_cache:
            path = os.path.join(self.root, 'movies.h5')
            h5file = tables.open_file(path, 'r')
            node = f'/movie{tgt["movie"]:03}_{tgt["segment"]:03}'
            movie = h5file.get_node(node)[:]
            movie_cache[index] = movie
            h5file.close()
        else:
            movie = movie_cache[index]

        assert tgt['start_frame'] >= 0 and tgt['end_frame'] <= movie.shape[0]

        # Movie segments are in the shape nframes x nchannels x ny x nx
        imgs = movie[tgt['start_frame']:tgt['end_frame'], ...].transpose((1, 0, 2, 3))
        
        # Center and normalize.
        # This seems like a random order, but it's to fit with the ordering
        # the standard ordering of conv3d. 
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html
        X = (imgs.astype(np.float32) - 
             np.array([83, 81, 73], dtype=np.float32).reshape((3, 1, 1, 1))) / 64.0
        mat_file = self.mat_files[tgt['key']]
        
        batch = mat_file['pepANA']['listOfResults'][tgt['result']]

        el = tgt['rel_electrode_num']
        y = []
        if batch['noRepeats'] > 1:
            y_ = 0
            for i in range(len(batch['repeat'])):
                # Bin the total number of spikes. This is simply the multi-unit activity.
                d_, _ = np.histogram(batch['repeat'][i]['data'][el][0], bins)
                y_ += d_
            y.append(y_ / float(len(batch['repeat'])))
        else:
            n_electrodes = len(batch['repeat']['data'])
            
            # Bin the total number of spikes. This is simply the multi-unit activity.
            d_, _ = np.histogram(batch['repeat']['data'][el][0], bins)
            y.append(d_)

        y = np.array(y).T.astype(np.float32)

        # Create a mask from the electrode range
        m = np.zeros((self.total_electrodes), dtype=np.bool)
        m[tgt['abs_electrode_num']] = True

        w = m * 1.0

        Y = np.zeros((self.total_electrodes, y.shape[0]))
        Y[tgt['abs_electrode_num'], :] = y.T

        return (X, m, w, Y)

    def __len__(self):
        # Returns the length of a dataset
        return len(self.sequence)

def _movie_info(root):
    """
    Build up a hashmap from tuples of (movie, segment) to info about the 
    movie, including location and duration
    """
    path = os.path.join(root, 'movies.h5')
    h5file = tables.open_file(path, 'r')
    movie_info = {}
    for i in range(30):
        for j in range(4):
            node = f'/movie{j:03}_{i:03}'
            nframes = len(h5file.get_node(node))
            
            movie_info[(j, i)] = {'nframes': nframes}
    
    h5file.close()
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