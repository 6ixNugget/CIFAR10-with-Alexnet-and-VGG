"""Routine for decoding the CIFAR-10 file format."""
import os
import sys
import tarfile
import urllib.request
import pickle
import numpy as np

IMAGE_SHAPE = (32,32)
IMAGE_CHANNEL = 3
IMAGE_BYTES = IMAGE_SHAPE[0] * IMAGE_SHAPE[1] * IMAGE_CHANNEL
NUM_TRAINING_RECORD = 50000
NUM_TEST_RECORD = 10000
NUM_CLASSES = 10

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'


def read_in_training_data(data_dir):
    """
    Read in training data from data_dir, default to DEFAULT_DATA_DIR.
    """
    data_dict = _read_in_data(get_training_batch_files(data_dir), NUM_TRAINING_RECORD)
    return data_dict["data"], data_dict["labels"]

def read_in_test_data(data_dir):
    """
    Read in test data from data_dir, default to DEFAULT_DATA_DIR.
    """
    data_dict = _read_in_data(get_test_batch_files(data_dir), NUM_TEST_RECORD)
    return data_dict["data"], data_dict["labels"]

def get_training_batch_files(data_dir):
    """
    Find cifar10 training data batch files under data_dir.
    """
    filenames = [os.path.join(os.getcwd(), data_dir, 'data_batch_%d' % i) for i in range(1, 6)]
    for file_path in filenames:
        if not os.path.isfile(file_path):
            raise ValueError('Failed to find file: ' + file_path)
    return filenames

def get_test_batch_files(data_dir):
    """
    Find cifar10 test data batch files under data_dir.
    """
    filenames = [os.path.join(os.getcwd(), data_dir, 'test_batch')]
    for file_path in filenames:
        if not os.path.isfile(file_path):
            raise ValueError('Failed to find file: ' + file_path)
    return filenames

def _read_in_data(filenames, expected_num_record):
    """
    Read in data from disk.
    Input:
        filenames: The path of data files.
    Output:
        data_dict: A dictionary object that contains all the images and its corresponding labels.
    """
    data_dict = dict()
    data_dict['labels'] = np.empty(0)
    data_dict['data'] = np.empty(shape=(0, *IMAGE_SHAPE, IMAGE_CHANNEL), dtype=np.float32)
    data_dict['filenames'] = np.empty(0)

    for file_path in filenames:
        new_dict = _unpickle(file_path)
        data_dict['labels'] = np.concatenate((data_dict['labels'], new_dict[b'labels']))
        data_dict['data'] = np.concatenate((data_dict['data'], 
                                            new_dict[b'data'].astype(np.float32)
                                                             .reshape(-1, IMAGE_CHANNEL, IMAGE_SHAPE[0]*IMAGE_SHAPE[1])
                                                             .transpose((0, 2, 1))
                                                             .reshape(-1, *IMAGE_SHAPE, IMAGE_CHANNEL)
                                            ))
        data_dict['filenames'] = np.concatenate((data_dict['filenames'], new_dict[b'filenames']))

    assert (len(data_dict['labels']) == expected_num_record and
            len(data_dict['data']) == expected_num_record and
            len(data_dict['filenames']) == expected_num_record), \
            "Number of data records does not match presets."

    return data_dict

def _unpickle(filename):
    """
    Unpickle a file using bytes encoding.
    """
    with open(filename, 'rb') as file:
        data_dict = pickle.load(file, encoding='bytes')
    return data_dict

def img_normalize(arr):
    """
    Zero mean unit variance image normalization.
    """
    mean = np.mean(arr, axis=(0, 1, 2, 3))
    std = np.std(arr, axis=(0, 1, 2, 3))
    arr = (arr - mean)/(std + 1e-7)
    return arr

def maybe_download_and_extract(dest_directory):
    """Download and extract the tarball from Alex's website."""
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    if not os.path.exists(filename):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, 
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filename, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall()