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
import librosa
import librosa.display
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
import warnings
warnings.filterwarnings('ignore')
import json
import pickle
from sklearn import preprocessing
from sklearn.model_selection import KFold, train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, classification_report, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


def save_pickle_files(file_, file_path):
    with open(file_path, 'wb') as f:
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




    ##RANDOM FOREST GRIDSEARCH
    random_forest_grid = {'max_depth': [None],
                        'max_features': ['sqrt', None],
                        'min_samples_split': [2, 4, 6],
                        'min_samples_leaf': [2, 4, 6],
                        'n_estimators': [100],
                        'random_state': [1]}

    rf_gridsearch = GridSearchCV(RandomForestClassifier(),
                                random_forest_grid,
                                #  n_jobs=-1,
                                verbose=True,
                                cv=2,
                                scoring='neg_log_loss'
                                )

    rf_gridsearch.fit(X_train, y_train)

    best_rf_model = rf_gridsearch.best_params_