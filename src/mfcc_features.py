import numpy as np
import scipy.io.wavfile as wav
from scipy.fftpack import dct
import librosa
import IPython.display as ipd
# %matplotlib inline
import matplotlib.pyplot as plt
import librosa.display
import soundfile
import sklearn
import shutil

import pandas as pd
import numpy as np
from numpy import argmax
import matplotlib.pyplot as plt
# %matplotlib inline
import librosa
import librosa.display
import IPython.display
import random
import warnings
import os
from PIL import Image
import pathlib
import csv
# sklearn Preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
#Keras
import keras
import warnings
warnings.filterwarnings('ignore')
from keras import layers
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling1D, AveragePooling2D, Input, Add
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical
from comet_ml import Experiment
import json

#to get mfcc features for a single file
def extract_mfcc_features(file_path):
    audio, sr = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed

#get mfcc features for all files
def get_all_features(df, audio_folder):
    features = []
    for item in df.index:
        filename = item + '.wav'
        data = extract_mfcc_features(f'{audio_folder}/{filename}')
        class_label = df.loc[item]['instrument_family_str']
        features.append([data, class_label, item])

    featuresdf = pd.DataFrame(features, columns=['feature', 'class_label', 'item'])

    featuresdf['class_label_num'] = featuresdf['class_label'].replace({
                                    'bass': 0,
                                    'brass': 1,
                                    'flute': 2,
                                    'guitar': 3,
                                    'keyboard': 4,
                                    'mallet': 5,
                                    'organ': 6,
                                    'reed': 7,
                                    'string': 8,
                                    'synth_lead': 9,
                                    'vocal': 10})
    featuresdf.to_csv('data/train_features_df.csv', index=None)
    return featuresdf

if __name__ == "__main__":

    #get training meta  data
    json_file_path = "/Users/katewessels/Documents/capstones/right_in_tune/data/nsynth-train/train_examples.json"
    with open(json_file_path, 'r') as j:
        contents = json.loads(j.read())
    df = pd.DataFrame(contents).T
    # df = pd.read_csv('data/training_metadata.csv')
    labels = list(df['instrument_family_str'].unique())
    labels[-1] = 'synth'

    #get mfcc features for all training audio files
    audio_folder = '/Users/katewessels/Documents/capstones/right_in_tune/data/nsynth-train/audio'

    #get featuresdf
    featuresdf = get_all_features(df, audio_folder)

    #read csv
    featuresdf = pd.read_csv('data/train_features_df.csv')

    #get feature/target numpy arrays
    X = np.array(featuresdf.feature.tolist())
    y = np.array(featuresdf.class_label_num.tolist())

    #split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=127)

    # #initial model
    # #params
    # num_labels = 11
    # batch_size = 32
    # num_epochs = 100
    # #instantiate
    # model = Sequential()
    # model.add(Flatten(input_shape=(40,)))
    # model.add(Dense(256))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(256))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(num_labels))
    # model.add(Activation('softmax'))
    # #compile
    # model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    # #summary
    # model.summary()

    # # Calculate pre-training accuracy
    # score = model.evaluate(X_test, y_test, verbose=0)
    # print("Pre-training Accuracy: {0:.2%}".format(score[1]))

    # #fit
    # model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs,
    #         validation_data = (X_test, y_test), verbose=2)

    # # Evaluate model on training and testing set
    # score = model.evaluate(x_train, y_train, verbose=0)
    # print("Training Accuracy: {0:.2%}".format(score[1]))
    # score = model.evaluate(x_test, y_test, verbose=0)
    # print("Testing Accuracy: {0:.2%}".format(score[1]))