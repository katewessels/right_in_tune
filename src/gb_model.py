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
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing
from sklearn.model_selection import KFold, train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, classification_report, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE

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
    X = np.array(featuresdf.feature.tolist())
    y = np.array(featuresdf.class_label_num.tolist())
    save_pickle_files(X, 'pickles/X.pkl')
    save_pickle_files(y, 'pickles/y.pkl')
    save_pickle_files(featuresdf, 'pickles/featuresdf.pkl')
    return featuresdf, X, y

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

    ## GRADIENT BOOSTING CLASSIFIER
    #fit
    model = GradientBoostingClassifier(learning_rate=0.1, max_depth=None, subsample=0.5,\
                                        min_samples_leaf=3, max_features='sqrt', n_estimators=100,\
                                        random_state=1)

    model.fit(X_train, y_train)

    #test predict
    y_pred = model.predict(X_train)
    y_pred_proba = model.predict_proba(X_test)
    #metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    #classification report
    report = classification_report(y_test, y_pred)

    #training subset predict
    train_y_pred = model.predict(X_train)
    train_y_pred_proba = model.predict_proba(X_train)
    train_accuracy = accuracy_score(y_train, train_y_pred)
    train_recall = recall_score(y_train, train_y_pred, average='macro')
    train_precision = precision_score(y_train, train_y_pred, average='macro')
    train_auc = roc_auc_score(y_train, train_y_pred_proba, multi_class='ovr')


    ###CROSS VALIDATE (default cv: stratified kfold (5-folds), which works for multi class)
    # precision_scores = cross_val_score(model, scaled_tfidf_matrix, y_train, scoring=make_scorer(precision_score, average='macro'))
    # accuracy_scores = cross_val_score(model, scaled_tfidf_matrix, y_train, scoring=make_scorer(accuracy_score))
    # recall_scores = cross_val_score(model, scaled_tfidf_matrix, y_train, scoring=make_scorer(recall_score, average='macro'))
    # auc_scores = cross_val_score(model, scaled_tfidf_matrix, y_train, scoring='roc_auc_ovr')

    # print(f'Training Mean CV Accuracy: {round(np.mean(accuracy_scores), 5)}')
    # print(f'Training Mean CV Precision: {round(np.mean(precision_scores), 5)}')
    # print(f'Training Mean CV Recall: {round(np.mean(recall_scores), 5)}')
    # print(f'Training Mean CV AUC Score: {round(np.mean(auc_scores), 5)}')

