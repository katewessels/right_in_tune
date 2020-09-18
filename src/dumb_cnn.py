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
from datetime import datetime
pd.set_option('display.max_columns', None)


def get_pickle_files(path_to_file):
    with open(path_to_file, 'rb') as f:
        file_ = pickle.load(f)
    return file_

def get_X_y(matrix, idx_list, df):
    '''
    reformat dataframe to correspond to indices of X, y.
    '''

    # X = np.zeros((matrix.shape[0], (matrix[0, :, :, 0].flatten().shape[0])))
    # for row_idx in range(matrix.shape[0]):
    #     X[row_idx] = list(rgb2gray(matrix[row_idx]).reshape(-1,))
    X = matrix
    series = pd.Series(idx_list, index=idx_list)
    concat_df = pd.concat([series, df], axis=1, sort=False, join='inner').drop(columns=[0])
    y = np.array(concat_df['instrument_family'])

    return concat_df, X, y


if __name__ == "__main__":

    #GET DATAFRAMES
    test_df = pd.read_csv('data/test_metadata.csv').set_index('note_str')
    train_df = pd.read_csv('data/training_metadata.csv').set_index('note_str')
    valid_df = pd.read_csv('data/valid_metadata.csv').set_index('note_str')

    #LOAD PICKLE FILES: IMAGE MATRIX AND IDX_LIST
    #test
    test_matrix = get_pickle_files('pickles/test_matrix.pkl')
    test_idx_list = get_pickle_files('pickles/test_idx_list.pkl')
    # #valid
    valid_matrix = get_pickle_files('pickles/valid_matrix.pkl')
    valid_idx_list = get_pickle_files('pickles/valid_idx_list.pkl')
    # #train
    train_matrix = get_pickle_files('pickles/train_matrix.pkl')
    train_idx_list = get_pickle_files('pickles/train_idx_list.pkl')

    #GET X AND y AND CONCAT_DF (DF W/ CORRESPONDING ROW INDICES)
    #test
    test_concat_df, X_test, y_test = get_X_y(test_matrix, test_idx_list, test_df)
    # #valid
    valid_concat_df, X_valid, y_valid = get_X_y(valid_matrix, valid_idx_list, valid_df)
    # #train
    train_concat_df, X_train, y_train = get_X_y(train_matrix, train_idx_list, train_df)


    # #DUMB CNN MODEL
    # #instantiate
    # model = keras.models.Sequential([
    #         keras.layers.Conv2D(16, (5, 5),
    #         activation='relu', input_shape=(300, 300, 1),
    #         padding='same'),
    #         keras.layers.MaxPooling2D((2,2))
    #         ])
    # model.add(keras.layers.Conv2D(32, (5, 5),
    #         activation='relu', padding='same'))
    # model.add(keras.layers.MaxPool2D(2,2))
    # model.add(keras.layers.Conv2D(32, (5, 5),
    #         activation='relu', padding='same'))
    # model.add(keras.layers.MaxPool2D(2,2))
    # model.add(keras.layers.Conv2D(32, (5, 5),
    #         activation='relu', padding='same'))
    # model.add(keras.layers.MaxPool2D(2,2))


    # model.add(keras.layers.Flatten())
    # model.add(keras.layers.Dense(128, activation='relu'))
    # model.add(keras.layers.Dense(11, activation='softmax'))
    # print(model.summary())
    # #compile
    # model.compile(optimizer='adam',
    #           loss='sparse_categorical_crossentropy',
    #           metrics=['accuracy'])
    # #fit
    # train_images = X_train.reshape((X_train.shape[0], 300, 300, 1))
    # train_labels = y_train
    # model.fit(train_images, train_labels, batch_size=32, epochs=5, verbose=2)

    # #test
    # test_images = X_test.reshape((X_test.shape[0], 300, 300, 1))
    # test_labels = y_test
    # test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    # #save model to an HDF5 file
    # saved_model_path = "./saved_models/cnn{}.h5".format(datetime.now().strftime("%m%d")) # _%H%M%S
    # model.save(saved_model_path)

    # #load/open a saved model
    # new_model = keras.models.load_model(saved_model_path)
    # print(new_model.summary())