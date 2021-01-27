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
import tensorflow.keras
import pickle
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling1D, GlobalAveragePooling2D, AveragePooling2D, Input, Add, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix #, plot_confusion_matrix
import itertools
from plot_models import plot_learning_curves, plot_confusion_matrix
import seaborn as sns
sns.set()
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.python.keras import backend as k


# CREATE BASE MODEL
def create_transfer_model(input_size, n_categories, weights='imagenet'):
        # base_model = keras.applications.xception.Xception(weights=weights,
        #                            include_top=False, input_shape=input_size) #layers=tf.keras.layers,
        base_model = Xception(weights=weights,
                            #   layers=tf.keras.layers,
                              include_top=False,
                              input_shape=input_size)

        model = base_model.output #added training=False param
        # model = base_model(base_model.input, training=False)
        model = GlobalAveragePooling2D()(model)
        # model = Dropout(0.5)(model) #added
        # model = Dense(1024, activation='relu')(model) #added
        predictions = Dense(n_categories, activation='softmax')(model)
        model = Model(inputs=base_model.input, outputs=predictions)
        return model

#DISPLAY MODEL LAYERS
def print_model_properties(model, indices = 0):
     for i, layer in enumerate(model.layers[indices:]):
        print("Layer {} | Name: {} | Trainable: {}".format(i+indices, layer.name, layer.trainable))

#CHANGE TRAINABLE LAYERS
def change_trainable_layers(model, trainable_index):
    for layer in model.layers[:trainable_index]:
        layer.trainable = False
    for layer in model.layers[trainable_index:]:
        layer.trainable = True

#PLOT MODEL PERFORMANCE
def plot_accuracy_loss(acc, val_acc, loss, val_loss, list_of_axvline_values, fig_num):
    plt.figure(figsize=(8, 8))
    #accuracy subplot
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    # plt.ylim([0, 1])
    plt.ylabel('Accuracy', fontsize=14)
    if list_of_axvline_values != []:
        for num in list_of_axvline_values:
            plt.axvline(num-1, color='green')
    plt.legend(fontsize=14)
    plt.title('Training and Validation Accuracy', fontsize=14)
    #loss subplot
    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    # plt.ylim([0, 1])
    plt.ylabel('Cross Entropy', fontsize=14)
    if list_of_axvline_values != []:
        for num in list_of_axvline_values:
            plt.axvline(num-1, color='green')
    plt.legend(fontsize=14)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.xlabel('epoch', fontsize=14)
    plt.savefig(f'images/transfer_layer{fig_num}.png', bbox_inches='tight')
    # plt.show()

#PICKLE FILES
def get_pickle_files(file_path):
    with open(file_path, 'rb') as f:
        file_ = pickle.load(f)
    return file_

def save_pickle_files(file_, file_path):
    with open(file_path, 'wb') as f:
        # Write the model to a file.
        pickle.dump(file_, f)
    return None

def update_metrics(metric_var, metric_str, history, if_first_model=False):
    if if_first_model:
        metric_var = []
    metric_var.extend(history.history[metric_str])
    save_pickle_files(metric_var, f'pickles/{metric_str}.pkl')
    return metric_var

def train_transfer_model(train_generator, valid_generator, optimizer, callbacks, unfreeze_layer, input_size, n_categories):
    #instantiate transfer model
    transfer_model = create_transfer_model(input_size, n_categories)
    print_model_properties(transfer_model)
    change_trainable_layers(transfer_model, unfreeze_layer)

    print_model_properties(transfer_model)
    #train model on new data, just top layer
    transfer_model.compile(optimizer=optimizer,
                loss=tf.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])
    history = transfer_model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=20,
        verbose=2,
        callbacks=callbacks)
    #get metrics
    list_of_axvline_values = []
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plot_accuracy_loss(acc, val_acc, loss, val_loss, list_of_axvline_values, unfreeze_layer)
    #pickle metrics and history (for next iteration)
    save_pickle_files(acc, 'pickles/accuracy.pkl')
    save_pickle_files(val_acc, 'pickles/val_accuracy.pkl')
    save_pickle_files(loss, 'pickles/loss.pkl')
    save_pickle_files(val_loss, 'pickles/val_loss.pkl')
    return history

def fine_tune_model(unfreeze_layer, last_unfreeze_layer, train_generator, valid_generator, optimizer, callbacks, initial_epochs, list_of_axvline_values, fine_tune_epochs=50):
    #get past metrics
    acc = get_pickle_files('pickles/accuracy.pkl')
    val_acc = get_pickle_files('pickles/val_accuracy.pkl')
    loss = get_pickle_files('pickles/loss.pkl')
    val_loss = get_pickle_files('pickles/val_loss.pkl')
    #set up epochs to accumulate for plotting
    total_epochs = initial_epochs + fine_tune_epochs
    #open saved transfer model
    transfer_model_filepath=f'saved_models/best_transfer_model_{last_unfreeze_layer}.hdf5'
    transfer_model = tensorflow.keras.models.load_model(transfer_model_filepath)
    print(transfer_model.summary())

    change_trainable_layers(transfer_model, unfreeze_layer) #update trainable layers, block by block

    print_model_properties(transfer_model)

    #fit new unfrozen layers
    transfer_model.compile(optimizer=optimizer,
                           loss=tf.losses.SparseCategoricalCrossentropy(),
                           metrics=['accuracy'])

    history = transfer_model.fit(
                            train_generator,
                            validation_data=valid_generator,
                            epochs=total_epochs,
                            initial_epoch = initial_epochs,
                            verbose=2,
                            callbacks=callbacks
                            )
    #update metrics now + pickle updated version (for next iteration)
    acc = update_metrics(acc, 'accuracy', history)
    val_acc = update_metrics(val_acc, 'val_accuracy', history)
    loss = update_metrics(loss, 'loss', history)
    val_loss = update_metrics(val_loss, 'val_loss', history)

    # plot
    plot_accuracy_loss(acc, val_acc, loss, val_loss, list_of_axvline_values, unfreeze_layer)
    return history

if __name__ == "__main__":

    #GET DATA GENERATORS (put this into a class)
    #params
    batch_size = 32
    img_height = 100
    img_width = 100
    num_classes = 11

    train_data_dir = pathlib.Path('data/nsynth-train/train_images')
    train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=preprocess_input,
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
    #test set: using test images
    test_data_dir = pathlib.Path('data/nsynth-test/test_images')
    test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=preprocess_input
                )
    test_generator = test_gen.flow_from_directory(
                directory=test_data_dir,
                seed=12,
                color_mode='rgb',
                target_size=(img_height, img_width),
                batch_size=1,
                shuffle=False,
                class_mode='sparse'
                )

    #TRANSFER LEARNING MODEL
    # INSTANTIATE AND TRAIN TOP LAYER WEIGHTS
    # unfreeze_layer = 132
    # optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1 = 0.9, beta_2 = 0.999)
    # callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,verbose=1),
    #             tf.keras.callbacks.ModelCheckpoint(filepath=f'saved_models/best_transfer_model_{unfreeze_layer}.hdf5',
    #                                     verbose=1, save_best_only=True, monitor='val_loss')]

    # history = train_transfer_model(train_generator, valid_generator, optimizer, callbacks, unfreeze_layer, (100, 100, 3), num_classes)



    # # #FINE TUNE MODEL, UNFREEZING ONE BLOCK OF LAYERS AT A TIME
    unfreeze_layer =  86 #update these with each iteration
    last_unfreeze_layer = 96 #update these with each iteration
    initial_epochs= 44 #update with each iteration (history.epoch[-1] at end of last iteration)
    list_of_axvline_values = [19, 26, 37, 44] #update w/ each iteration (append the initial_epoch)

    optimizer = tf.keras.optimizers.Adam(lr=0.0001, beta_1 = 0.9, beta_2 = 0.999)
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,verbose=1),
                tf.keras.callbacks.ModelCheckpoint(filepath=f'saved_models/best_transfer_model_{unfreeze_layer}.hdf5',
                                        verbose=1, save_best_only=True, monitor='val_loss')]


    history = fine_tune_model(unfreeze_layer, last_unfreeze_layer, train_generator, valid_generator, optimizer, callbacks, initial_epochs, list_of_axvline_values, fine_tune_epochs=50)



    # #OPEN A SAVED MODEL PATH
    # saved_model_path = f"./saved_models/best_transfer_model_{unfreeze_layer}.hdf5"
    # model = tf.keras.models.load_model(saved_model_path)
    # print(new_model.summary())

    # #EVALUATE MODEL ON TEST SET
    # test_loss, test_accuracy = model.evaluate(test_generator, verbose=2)
    # print(f"test_loss = {test_loss}, test_accuracy = {test_accuracy}")
    # y_probas = model.predict(test_generator)
    # y_preds = np.argmax(y_probas, axis=1)
    # y_actual = test_generator.classes

    # #compute confusion matrix
    # cm = confusion_matrix(y_actual, y_preds)
    # np.set_printoptions(precision=2)

    # #plot confusion matrix
    # class_names = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet',
    #                 'organ', 'reed', 'string', 'synth_lead', 'vocal']
    # plot_confusion_matrix(cm, classes=class_names, title='Confusion Matrix', image_name='trial13')
