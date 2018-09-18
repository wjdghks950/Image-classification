""" Preprocess and augment the CIFAR training data"""

import tensorflow as tf
import tarfile
import cv2
import os
import numpy as np
import pickle
from random import shuffle
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm


IMG_SIZE = 32 #CIFAR data image will be 30 x 30; any changes to the data dimension can be addressed by changing IMG_SIZE

class DownloadProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def load_label_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def DataLoader(datapath, batch_idx):
# For loading data for the RadarNet model
    print(os.path.exists(datapath))
    if os.path.exists(datapath):
        try:
            with open(datapath + '/data_batch_' + str(batch_idx), mode='rb') as file:
                batch = pickle.load(file, encoding='latin1')

            batch_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
            batch_labels = batch['labels']

        except IOError:
            print('Failed to read from {}'.format(datapath))

    return batch_features, batch_labels

def DownloadData(datapath):
    # If dataset is not already downloaded yet, download it
    print("Checking for downloaded dataset...")

    if not isfile('./CIFAR_Train_Data/cifar-10-python.tar.gz'):
        with DownloadProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as progressbar:
            urlretrieve('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
                        './CIFAR_Train_Data/cifar-10-python.tar.gz', progressbar.hook)

    if not isdir(datapath):
        with tarfile.open('./CIFAR_Train_Data/cifar-10-python.tar.gz') as tar:
            tar.extractall()
            tar.close()

def batch_features_labels(features, labels, batch_size):
    # Split features and labels into batches

    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]

def load_preprocess_training_batch(batch_id, batch_size):
    # Load the preprocessed Training data and return them in batches of batch_size or less
    filename = 'preprocess_batch' + str(batch_id) + '.p'
    features, labels = pickle.load(open(filename, mode='rb'))

    return batch_features_labels(features, labels, batch_size)

def display_stats(datapath, batch_idx, sample_idx):
    features, labels = DataLoader(datapath, batch_idx)

    if not(0 <= sample_idx < len(features)):
        print('{} samples in batch {}. {} is out of range.'.format(len(features), batch_idx, sample_idx))

    print('\nStats of batch #{}:'.format(batch_idx))
    print('# of Samples: {}\n'.format(len(features)))

    label_names = load_label_names()
    label_count = dict(zip(*np.unique(labels, return_counts=True)))

    for key, value in label_count.items():
        print('Label Counts of [{}]({}) : {}'.format(key, label_names[key].upper(), value))

    sample_img = features[sample_idx]
    sample_lbl = labels[sample_idx]

    print('\nExample of Image {}:'.format(sample_idx))
    print('Image - Min Value: {} Max Value: {}'.format(sample_img.min(), sample_img.max()))
    print('Image - Shape: {}'.format(sample_img.shape))
    print('Label - Label Id: {} Name: {}'.format(sample_lbl, label_names[sample_lbl]))

def normalize(x):
    #Using min-max normalization to normalize the input data

    min_val = np.min(x)
    max_val = np.max(x)

    x = (x-min_val) / (max_val - min_val)
    return x

def one_hot_encode(x):
    encoded = np.zeros((len(x), 10))

    for idx, val in enumerate(x):
        encoded[idx][val] = 1

    return encoded

def _preprocess_and_save(normalize, one_hot_encode, features, labels, filename):
    features = normalize(features)
    labels = one_hot_encode(labels)

    pickle.dump((features, labels), open(filename, 'wb'))

def preprocess_and_save_data(datapath, normalize, one_hot_encode):
    num_batches = 5
    valid_features = []
    valid_labels = []

    for batch_idx in range(1, num_batches + 1):
        features, labels = DataLoader(datapath, batch_idx)

        valid_idx = int(len(features) * 0.1) #index which determines the end of validation set (10%)

        # Preprocess the rest of 90% of training data of the ith batch
        _preprocess_and_save(normalize, one_hot_encode, features[:-valid_idx], labels[:-valid_idx], 'preprocess_batch' + str(batch_idx) + '.p')

        valid_features.extend(features[-valid_idx:])
        valid_labels.extend(labels[-valid_idx:])

        # Preprocess all the stacked validation dataset
        _preprocess_and_save(normalize, one_hot_encode, np.array(valid_features), np.array(valid_labels), 'preprocess_validation.p')

        #load test dataset
        if os.path.exists(datapath):
            try:
                with open(datapath + '/test_batch', mode='rb') as file:
                    batch = pickle.load(file, encoding='latin1')
            except IOError:
                print('Error occurred whlie opening the following file:{}'.format(TEST_DIR + '/test_batch'))

        test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
        test_labels = batch['labels'] 

        _preprocess_and_save(normalize, one_hot_encode, np.array(test_features), np.array(test_labels), 'preprocess_training.p')
