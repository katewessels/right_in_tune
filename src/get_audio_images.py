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

def move_images_to_class_folders(image_folder, file_limit=False):
    for filename in os.listdir(image_folder):
        idx = filename.find('_')
        class_ = os.path.basename(filename)[:idx]
        if not os.path.isdir(os.path.join(image_folder, class_)):
            os.mkdir(os.path.join(image_folder, class_))
        if file_limit==False:
            shutil.move(os.path.join(image_folder,filename), os.path.join(image_folder, class_))
        #set a file limit to balance classes in training set
        else:
            num_files = len(os.listdir(os.path.join(image_folder, class_)))
            if num_files < 2000:
                shutil.move(os.path.join(image_folder,filename), os.path.join(image_folder, class_))

def flatten_directory(dir_to_flatten):
    for dirpath, dirnames, filenames in os.walk(dir_to_flatten):
        for filename in filenames:
            try:
                os.rename(os.path.join(dirpath, filename), os.path.join(dir_to_flatten, filename))
            except OSError:
                print ("Could not move %s " % os.path.join(dirpath, filename))


if __name__ == "__main__":
    #GET MFCC IMAGES FROM AUDIO FILES
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

    #MOVE MFCC IMAGES INTO LABELED CLASS FOLDERS
    #train
    train_image_folder = '/Users/katewessels/Documents/capstones/right_in_tune/data/nsynth-train/train_images'
    move_images_to_class_folders(train_image_folder, file_limit=True)
    #valid
    valid_image_folder = '/Users/katewessels/Documents/capstones/right_in_tune/data/nsynth-valid/valid_images'
    move_images_to_class_folders(valid_image_folder)
    #test
    test_image_folder = '/Users/katewessels/Documents/capstones/right_in_tune/data/nsynth-test/test_images'
    move_images_to_class_folders(test_image_folder)


