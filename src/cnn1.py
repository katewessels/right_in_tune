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
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization, Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling1D, AveragePooling2D, Input, Add
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
import itertools
import seaborn as sns
sns.set()
import tensorflow as tf
from tensorflow.keras import backend as k #resolving tensorflow issues within ec2 instance
from tensorflow.python.keras import backend as k #resolving tensorflow issues within ec2 instance
from plot_models import plot_learning_curves, plot_confusion_matrix


# Image Data Generators
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
            color_mode='rgb',
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
            color_mode='rgb',
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='sparse',
            shuffle=True
            )

#test set: using a sample from the "validation folder"- these are unseen images
test_data_dir = pathlib.Path('data/nsynth-valid/valid_images')
test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale= 1./255,
            validation_split=.17 #take only 17% of the 12,000 total files for testing
            )
test_generator = test_gen.flow_from_directory(
            directory=test_data_dir,
            seed=12,
            subset='validation',
            color_mode='rgb',
            target_size=(img_height, img_width),
            batch_size=1,
            shuffle=False,
            class_mode='sparse'
            )


##TRAIN CNN
#params
num_classes = 11
num_channels = 3

#architecture 1
model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, 3, input_shape=(img_height, img_width, num_channels),
                            activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
        ])


#train model
optimizer = tf.keras.optimizers.Adam(lr=0.0001, beta_1 = 0.9, beta_2 = 0.999)

model.compile(optimizer=optimizer,
              loss=tf.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

callbacks = [
    tf.keras.callbacks.EarlyStopping(
            # Stop training when `val_loss` is no longer improving
            monitor="val_loss",
            patience=20,
            verbose=1
            )]

history = model.fit(
            train_generator,
            validation_data=valid_generator,
            epochs=100,
            verbose=2 #,
            # callbacks=callbacks
            )

print(model.summary())

#save model to an HDF5 file
saved_model_path = "./saved_models/cnn_arc1_13.h5"
model.save(saved_model_path)

# #to open a saved model
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
fig, ax = plt.subplots(figsize=(15,7))
plot_learning_curves(history, 'trial13', ax=ax)

#compute confusion matrix
cm = confusion_matrix(y_actual, y_preds)
np.set_printoptions(precision=2)

#plot confusion matrix
class_names = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet',
                'organ', 'reed', 'string', 'synth_lead', 'vocal']
plot_confusion_matrix(cm, classes=class_names, title='Confusion Matrix', image_name='trial13')
