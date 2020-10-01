import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import librosa
import IPython.display as ipd
import librosa.display
import soundfile
import sklearn
import random
import os
from PIL import Image
import pathlib
import csv
import scipy.io.wavfile as wavfile
from scipy.fftpack import dct
from sklearn.model_selection import train_test_split
import keras
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
from keras import layers
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling1D, AveragePooling2D, Input, Add
from keras.models import Sequential
from keras.optimizers import SGD
from skimage import io, color, filters
from skimage.transform import resize, rotate
from skimage.color import rgb2gray
import pickle

pd.set_option('display.max_columns', None)

def get_image_matrix(path_to_image_folder):
    matrix = []
    idx_list = []
    for filename in os.listdir(path_to_image_folder):
        image = io.imread(f'{path_to_image_folder}/{filename}')/255
        exp_image = np.expand_dims(image, axis=0)
        idx_list.append(f"{filename[:-4]}")
        if matrix == []:
            matrix = exp_image
        else:
            matrix = np.append(matrix, exp_image, axis=0)
            if matrix.shape[0] == 1000:
                break
    return matrix, idx_list

def save_pickle_files(file_, file_path):
    with open(file_path, 'wb') as f:
        # Write the model to a file.
        pickle.dump(file_, f)
    return None

if __name__ == "__main__":

    #GET DATAFRAMES
    test_df = pd.read_csv('data/test_metadata.csv').set_index('note_str')
    train_df = pd.read_csv('data/training_metadata.csv').set_index('note_str')
    valid_df = pd.read_csv('data/valid_metadata.csv').set_index('note_str')

    #GET IMAGE TENSORS (MATRICES) AND IDX_LISTS AND PICKLE
    #test set
    test_matrix, test_idx_list = get_image_matrix('data/nsynth-test/test_images')
    save_pickle_files(test_matrix, 'pickles/test_matrix.pkl')
    save_pickle_files(test_idx_list, 'pickles/test_idx_list.pkl')
    #valid set
    valid_matrix, valid_idx_list = get_image_matrix('data/nsynth-valid/valid_images')
    save_pickle_files(valid_matrix, 'pickles/valid_matrix.pkl')
    save_pickle_files(valid_idx_list, 'pickles/valid_idx_list.pkl')
    #train set
    train_matrix, train_idx_list = get_image_matrix('data/nsynth-train/train_images')
    save_pickle_files(train_matrix, 'pickles/train_matrix.pkl')
    save_pickle_files(train_idx_list, 'pickles/train_idx_list.pkl')




