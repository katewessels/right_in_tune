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
# from tensorflow.keras import layers
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




def get_pickle_files(path_to_file):
    with open(path_to_file, 'rb') as f:
        file_ = pickle.load(f)
    return file_

def get_X_y(matrix, idx_list, df):
    '''
    reshape image data and take grayscale for X
    y is target
    '''

    X = np.zeros((matrix.shape[0], (matrix[0, :, :, 0].flatten().shape[0])))
    for row_idx in range(matrix.shape[0]):
        X[row_idx] = list(rgb2gray(matrix[row_idx]).reshape(-1,))

    series = pd.Series(idx_list, index=idx_list)
    concat_df = pd.concat([series, df], axis=1, sort=False, join='inner').drop(columns=[0])
    y = np.array(concat_df['instrument_family'])

    return concat_df, X, y




if __name__ == "__main__":

    #GET DATAFRAMES
    test_df = pd.read_csv('data/test_metadata.csv').set_index('note_str')
    train_df = pd.read_csv('data/training_metadata.csv').set_index('note_str')
    valid_df = pd.read_csv('data/valid_metadata.csv').set_index('note_str')

    #LOAD PICKLE FILES
    #test
    test_matrix = get_pickle_files('pickles/test_matrix.pkl')
    test_idx_list = get_pickle_files('pickles/test_idx_list.pkl')
    # #valid
    # valid_matrix = get_pickle_files('pickles/valid_matrix.pkl')
    # valid_idx_list = get_pickle_files('pickles/valid_idx_list.pkl')
    # #train
    # train_matrix = get_pickle_files('pickles/train_matrix.pkl')
    # train_idx_list = get_pickle_files('pickles/train_idx_list.pkl')

    #GET X AND y
    #test
    test_concat_df, X_test, y_test = get_X_y(test_matrix, test_idx_list, test_df)
    # #valid
    # valid_concat_df, X_valid, y_valid = get_X_y(valid_matrix, valid_idx_list, valid_df)
    # #train
    # train_concat_df, X_train, y_train = get_X_y(train_matrix, train_idx_list, train_df)



    # #to get corresponding instrument names to the instrument class labels in y
    # for i in range(len(y_test)):
    #     print(concat_test_df['instrument_family_str'][i])

