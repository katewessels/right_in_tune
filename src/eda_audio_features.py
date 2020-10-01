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
from numpy import argmax
import matplotlib.pyplot as plt
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
import json
import seaborn as sns
sns.set()
pd.set_option('display.max_columns', None)

#to get 40 mfcc features for a single file
def extract_mfcc_features(file_path):
    audio, sr = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed

#to get 12 mfcc features for a single file
def extract_12_mfcc_features(file_path):
    audio, sr = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=12)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed

if __name__ == "__main__":

    #get training set audio meta data
    json_file_path = "/Users/katewessels/Documents/capstones/right_in_tune/data/nsynth-train/train_examples.json"
    with open(json_file_path, 'r') as j:
        contents = json.loads(j.read())

    #geta audio data into a dataframe
    df = pd.DataFrame(contents).T
    #class labels
    labels = list(df['instrument_family_str'].unique())
    labels[-1] = 'synth'

    #get a random sample from each class
    files = dict()
    audio_folder = '/Users/katewessels/Documents/capstones/right_in_tune/data/nsynth-train/audio'

    for filename in os.listdir(audio_folder):
        idx = filename.find('_')
        for label in labels:
            if os.path.basename(filename)[:idx] == label:
                path_to_file = f'{audio_folder}/{filename}'
                files[label] = path_to_file

    #plot example waveform from each class
    fig = plt.figure(figsize=(15, 9))
    fig.subplots_adjust(hspace=0.5, wspace=0.25)
    for i, label, in enumerate(labels):
        file_path = files[label]
        fig.add_subplot(3, 4, i+1)
        plt.title(label)
        audio, sr = librosa.load(file_path)
        librosa.display.waveplot(audio, sr=sr, alpha=0.7)
    plt.savefig('images/class_waveforms.png', bbox_inches='tight')

    #plot example mfcc from each class
    fig = plt.figure(figsize=(20, 12))
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    for i, label, in enumerate(labels):
        file_path = files[label]
        fig.add_subplot(3, 4, i+1)
        plt.title(label)
        audio, sr = librosa.load(file_path)
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel',x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout
    plt.savefig('images/class_melspecs.png', bbox_inches='tight')

    #plot example mfcc from each class
    fig = plt.figure(figsize=(20, 12))
    fig.subplots_adjust(hspace=0.5, wspace=0.4)
    for i, label, in enumerate(labels):
        file_path = files[label]
        fig.add_subplot(3, 4, i+1)
        plt.title(label)
        audio, sr = librosa.load(file_path)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=12)
        librosa.display.specshow(mfccs, sr=sr, x_axis='time')
        plt.ylabel('MFC Coefficients')
        plt.colorbar()
        plt.yticks(np.arange(0, 12), labels=None)
    plt.savefig('images/class_mfccs.png', bbox_inches='tight')

    #example- get mfcc features from a single audio file
    mfcc_features_example = extract_12_mfcc_features('/Users/katewessels/Documents/capstones/right_in_tune/data/nsynth-train/audio/keyboard_acoustic_016-022-100.wav')
