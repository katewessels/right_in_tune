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
import PIL
import PIL.Image
import tensorflow as tf
import glob
import shutil
import keras
from keras import layers
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling1D, AveragePooling2D, Input, Add
from keras.models import Sequential
from keras.optimizers import SGD
from datetime import datetime
from dumb_cnn import get_pickle_files, get_X_y
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix #, plot_confusion_matrix
import itertools
from plot_models import plot_learning_curves, plot_confusion_matrix
import seaborn as sns
sns.set()

#GET DATA GENERATORS
#params
batch_size = 32
img_height = 180
img_width = 180
#train set
train_data_dir = pathlib.Path('data/nsynth-train/train_images')
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
            )
train_generator = train_gen.flow_from_directory(
    directory=train_data_dir,
    seed=52,
    subset='training',
    color_mode='rgba',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse',
    shuffle=True
    )
#valid set
valid_generator = train_gen.flow_from_directory(
    directory=train_data_dir,
    seed=52,
    subset='validation',
    color_mode='rgba',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse',
    shuffle=True
    )
#test set: using test images
test_data_dir = pathlib.Path('data/nsynth-test/test_images')
test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255
            )
test_generator = test_gen.flow_from_directory(
    directory=test_data_dir,
    seed=12,
    color_mode='rgba',
    target_size=(img_height, img_width),
    batch_size=1,
    shuffle=False,
    class_mode='sparse'
    )

# #another test set (using validation images)
# test_data_dir = pathlib.Path('data/nsynth-valid/valid_images')
# test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
#             rescale=1./255
#             )
# test_generator = test_gen.flow_from_directory(
#     directory=test_data_dir,
#     seed=1234,
#     color_mode='rgba',
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     shuffle=False,
#     class_mode='sparse'
#     )


##TRAIN MODEL
num_classes = 11

model = keras.models.Sequential([
        # keras.layers.experimental.preprocessing.Rescaling(1./255), #only if using datasets from directories
        #keras.layers.experimental.preprocessing.Resizing(img_height, img_width),
        keras.layers.Conv2D(32, 3, activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, 3, activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, 3, activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, 3, activation='relu'),
        keras.layers.MaxPooling2D(),
        # keras.layers.Dropout(0.5),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation='softmax')
        ])
#compile
optimizer = tf.keras.optimizers.Adam(lr=0.0001, beta_1 = 0.9, beta_2 = 0.999)
model.compile(optimizer=optimizer,
              loss=tf.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

#early stopping condition
callbacks = [
    keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor="val_loss",
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-2,
        # "no longer improving" being further defined as "for at least 3 epochs"
        patience=3,
        verbose=1,
    )]

#fit
history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=2,
    verbose=2,
    callbacks=callbacks
    )

print(model.summary())

#save model to an HDF5 file
saved_model_path = "./saved_models/cnn_with_train5{}.h5".format(datetime.now().strftime("%m%d")) # _%H%M%S
# model.save(saved_model_path)

# #load/open a saved model
# saved_model_path = "./saved_models/cnn20917.h5"
# new_model = keras.models.load_model(saved_model_path)
# print(new_model.summary())

#EVALUATE MODEL ON TEST SET
test_loss, test_accuracy = model.evaluate(test_generator, verbose=2)
print(f"test_loss = {test_loss}, test_accuracy = {test_accuracy}")
y_probas = model.predict(test_generator)
y_preds = np.argmax(y_probas, axis=1)
y_actual = test_generator.classes

#PLOT RESULTS
#plot learning curves
fig, ax = plt.subplots(figsize=(8, 5))
plot_learning_curves(history, 'trial2', ax=ax)

# Compute confusion matrix
cm = confusion_matrix(y_actual, y_preds)
np.set_printoptions(precision=2)

# Plot confusion matrix
class_names = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet',
                'organ', 'reed', 'string', 'synth_lead', 'vocal']
plot_confusion_matrix(cm, classes=class_names,
                      title='Confusion Matrix')
