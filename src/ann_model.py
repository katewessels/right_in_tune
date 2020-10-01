import numpy as np
import scipy.io.wavfile as wav
from scipy.fftpack import dct
import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa.display
import soundfile
import sklearn
import shutil
import pandas as pd
import numpy as np
from numpy import argmax
import IPython.display
import random
import warnings
import os
from PIL import Image
import pathlib
import csv
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
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
import pickle
from imblearn.over_sampling import SMOTE


def save_pickle_files(file_, file_path):
    with open(file_path, 'wb') as f:
        # Write the model to a file.
        pickle.dump(file_, f)
    return None

def get_pickle_files(file_path):
    with open(file_path, 'rb') as f:
        file_ = pickle.load(f)
    return file_

if __name__ == "__main__":
    #get mfcc feature data
    X = get_pickle_files('pickles/X.pkl')
    y = get_pickle_files('pickles/y.pkl')
    featuresdf = get_pickle_files('pickles/featuresdf.pkl')

    #split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=127)

    #set up test and train sets first, which we already have
    sm = SMOTE(random_state=462)
    # print('Original dataset shape %s' % Counter(y_train))
    X_train, y_train = sm.fit_sample(X_train, y_train)
    # print('Resampled dataset shape %s' % Counter(y_train))

    #initial model
    #params
    num_labels = 11
    batch_size = 32
    num_epochs = 100
    #instantiate
    model = Sequential()
    model.add(Flatten(input_shape=(40,)))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))
    #compile
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    #summary
    model.summary()

    # Calculate pre-training accuracy
    score = model.evaluate(X_test, y_test, verbose=0)
    print("Pre-training Accuracy: {0:.2%}".format(score[1]))

    #fit
    model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs,
            validation_data = (X_test, y_test), verbose=2)


    #pickle model
    save_pickle_files(model, 'pickles/ann_model.pkl')

    # Evaluate model on training and testing set
    score = model.evaluate(X_train, y_train, verbose=0)
    print("Training Accuracy: {0:.2%}".format(score[1]))
    score = model.evaluate(X_test, y_test, verbose=0)
    print("Testing Accuracy: {0:.2%}".format(score[1]))