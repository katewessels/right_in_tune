import numpy as np
import pandas as pd
from numpy import argmax
import scipy.io.wavfile as wavfile
from scipy.fftpack import dct
import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa.display
import soundfile
import sklearn
import IPython.display as ipd
import random
import warnings
import os
from PIL import Image
import pathlib
import csv
import scipy.io.wavfile as wavfile
from scipy.fftpack import dct


def get_mfccs(path_to_audio_folder, path_to_image_folder):
    for filename in os.listdir(path_to_audio_folder):
        audio_file_path = f'{path_to_audio_folder}/{filename}'
        # x, sr = librosa.load(audio_file_path, sr=16000)
        x, sr = librosa.load(str(pathlib.Path(audio_file_path)))
        mfccs = librosa.feature.mfcc(x, sr=sr)
        librosa.display.specshow(mfccs, sr=sr, x_axis='time')
        plt.axis('off')
        plt.savefig(f"{path_to_image_folder}/{filename[:-3].replace('.', '')}.png")
        plt.clf()
    return None

def count_images(path_to_audio_folder, path_to_image_folder):
    audio_count = 0
    image_count = 0
    for filename in os.listdir(path_to_audio_folder):
        audio_count += 1
    for filename in os.listdir(path_to_image_folder):
        image_count += 1
    return audio_count, image_count


if __name__ == "__main__":

    #create image folder "train_images"
    pathlib.Path('data/nsynth-train/train_images').mkdir(parents=True, exist_ok=True)
    #get train mfcc images in "train_images" folder
    get_mfccs('data/nsynth-train/audio', 'data/nsynth-train/train_images')
    # train_audio_count, train_image_count = count_images('data/nsynth-train/audio', 'data/nsynth-train/train_images')

    #create image folder "valid_images"
    pathlib.Path('data/nsynth-valid/valid_images').mkdir(parents=True, exist_ok=True)
    # get valid mfcc images in "valid_images" folder
    get_mfccs('data/nsynth-valid/audio', 'data/nsynth-valid/valid_images')
    # valid_audio_count, valid_image_count = count_images('data/nsynth-valid/audio', 'data/nsynth-valid/valid_images')

    #create image folder "test_images"
    pathlib.Path('data/nsynth-test/test_images').mkdir(parents=True, exist_ok=True)
    #get test mfcc images in "test_images" folder
    get_mfccs('data/nsynth-test/audio', 'data/nsynth-test/test_images')
    # test_audio_count, test_image_count = count_images('data/nsynth-test/audio', 'data/nsynth-test/test_images')
