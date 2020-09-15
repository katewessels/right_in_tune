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



    # #write matrix and idx_list to a pickle file
    # with open('test_matrix.pkl', 'wb') as f:
    #     # Write the model to a file.
    #     pickle.dump(test_matrix, f)
    # with open('test_idx_list.pkl', 'wb') as f:
    #     # Write the model to a file.
    #     pickle.dump(test_idx_list, f)




    # # open pickle file
    # with open('test_matrix.pkl', 'rb') as f:
    #     test_matrix = pickle.load(f)
    # with open('test_idx_list.pkl', 'rb') as f:
    #     test_idx_list = pickle.load(f)


    # #train set
    # train_matrix, train_idx_list = get_image_matrix('data/nsynth-train/train_images')
    # #write matrix and idx_list to a pickle file
    # with open('train_matrix.pkl', 'wb') as f:
    #     # Write the model to a file.
    #     pickle.dump(train_matrix, f)
    # with open('train_idx_list.pkl', 'wb') as f:
    #     # Write the model to a file.
    #     pickle.dump(train_idx_list, f)

    # # open pickle file
    # with open('train_matrix.pkl', 'rb') as f:
    #     train_matrix = pickle.load(f)
    # with open('train_idx_list.pkl', 'rb') as f:
    #     train_idx_list = pickle.load(f)

    # #valid set
    # valid_matrix, valid_idx_list = get_image_matrix('data/nsynth-valid/valid_images')
    # #write matrix and idx_list to a pickle file
    # with open('valid_matrix.pkl', 'wb') as f:
    #     # Write the model to a file.
    #     pickle.dump(valid_matrix, f)
    # with open('valid_idx_list.pkl', 'wb') as f:
    #     # Write the model to a file.
    #     pickle.dump(valid_idx_list, f)


    # # open pickle file
    # with open('valid_matrix.pkl', 'rb') as f:
    #     valid_matrix = pickle.load(f)
    # with open('valid_idx_list.pkl', 'rb') as f:
    #     valid_idx_list = pickle.load(f)



    #GET IMAGES
    # path_to_image_folder = 'data/nsynth-train/train_images'
    # for filename in os.listdir(path_to_image_folder):
    # #     image = io.imread(f'{path_to_image_folder}/{filename}')
    # path_to_image_folder = 'data/nsynth-test/test_images'

    # filename = 'vocal_synthetic_003-108-127.png'
    # image = io.imread(f'{path_to_image_folder}/{filename}')
    # print(type(image))
    # print(image.shape)
    # # io.imshow(image)

    # image_gray = rgb2gray(image)
    # print(image_gray.shape)
    # print(image_gray.min())
    # print(image_gray.max())
    # # io.imshow(image_gray)

    # image_gray_values = np.ravel(image_gray)
    # print(image_gray_values.shape)

    # fig = plt.figure(figsize=(6,6))
    # ax = fig.add_subplot(111)
    # ax.hist(image_gray_values, bins=256)
    # ax.set_xlabel('pixel intensities', fontsize=14)
    # ax.set_ylabel('frequency in image', fontsize=14)
    # ax.set_title("MFCC image histogram", fontsize=16)
    # # plt.show()

    # In [4]: run src/cnn.py
    # <class 'numpy.ndarray'>
    # (700, 1500, 4)
    # (700, 1500)
    # 0.17196627450980392
    # 1.0
    # (1050000,)

    # (M, N, 4): an image with RGBA values
    # (0-1 float or 0-255 int),
    # an alpha channel, specifying opacity i.e. including transparency.


    #want a training image matrix
    #289,205 rows (images)
    #each row (image) is 700 x 1500 pixels
    #so a train_image_matrix.shape = (289205, 700, 1500, 4)
    #or flatten (and grayscale) to (289205, 1050000)
    #make df. set index to file_name (without.png)
    #then can merge train_df['instrument_family'] on index
    #keep y column ['instrument_family'] as target array
    #keep matrix image/pixel columns as training array

    #test
    # filename_1 = 'vocal_synthetic_003-108-127.png'
    # image_1 = io.imread(f'{path_to_image_folder}/{filename_1}')/255
    # filename_2 = 'vocal_synthetic_003-108-075.png'
    # image_2 = io.imread(f'{path_to_image_folder}/{filename_2}')/255

    # images = np.array([image_1, image_2])
    # batch_size, height, width, channels = images.shape

    # #create 2 filters
    # filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
    # filters[:, 3, :, 0] = 1 #vertical line
    # filters[3, :, :, 1] = 1 #horizontal line

    # outputs = tf.nn.conv2d(images, filters, strides=1, padding='SAME')
    # plt.imshow(outputs[0, :, :, 1], cmap='gray') #plot 1st image's 2nd feature map
    # # plt.show()

    #test adding images to matrix
    # image_1 = np.expand_dims(image_1, axis=0)
    # image_2 = np.expand_dims(image_2, axis=0)
    # matrix = np.append(image_1, image_2, axis=0)




