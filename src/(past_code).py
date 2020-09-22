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

def move_images_to_class_folders(image_folder, file_limit=False):
    for filename in os.listdir(image_folder):
        idx = filename.find('_')
        class_ = os.path.basename(filename)[:idx]
        if not os.path.isdir(os.path.join(image_folder, class_)):
            os.mkdir(os.path.join(image_folder, class_))
        #if no file limit, move all files
        if file_limit==False:
            shutil.move(os.path.join(image_folder,filename), os.path.join(image_folder, class_))
        #if file limit, only move a certain # of files into each folder (balanced classes)
        else:
            num_files = len(os.listdir(os.path.join(image_folder, class_)))
            if num_files < 4000:
                shutil.move(os.path.join(image_folder,filename), os.path.join(image_folder, class_))



# GET ALL TRAIN IMAGE FILES INTO CLASS LABELED FOLDERS
# image_folder = '/Users/katewessels/Documents/capstones/right_in_tune/data/nsynth-train/train_images'
# image_folder = 'data/nsynth-train/train_images'
# for filename in os.listdir(pathlib.Path(image_folder)):
#     idx = filename.find('_')
#     class_ = os.path.basename(filename)[:idx]
#     if not os.path.isdir(os.path.join(image_folder, class_)):
#         os.mkdir(os.path.join(image_folder, class_))
#     shutil.move(os.path.join(image_folder,filename), os.path.join(image_folder, class_))


#GET IMAGE FILES INTO CLASS LABELED FOLDERS
# image_folder = '/Users/katewessels/Documents/capstones/right_in_tune/data/nsynth-valid/valid_images'
# for filename in os.listdir(image_folder):
#     idx = filename.find('_')
#     class_ = os.path.basename(filename)[:idx]
#     if not os.path.isdir(os.path.join(image_folder, class_)):
#         os.mkdir(os.path.join(image_folder, class_))
#     shutil.move(os.path.join(image_folder,filename), os.path.join(image_folder, class_))

# test_image_folder = '/Users/katewessels/Documents/capstones/right_in_tune/data/nsynth-test/test_images'
# for filename in os.listdir(test_image_folder):
#     idx = filename.find('_')
#     class_ = os.path.basename(filename)[:idx]
#     if not os.path.isdir(os.path.join(test_image_folder, class_)):
#         os.mkdir(os.path.join(test_image_folder, class_))
#     shutil.move(os.path.join(test_image_folder,filename), os.path.join(test_image_folder, class_))

# train_image_folder = '/Users/katewessels/Documents/capstones/right_in_tune/data/nsynth-train/train_images'
# # move only 2000 samples into each class for balanced training
# for filename in os.listdir(train_image_folder):
#     idx = filename.find('_')
#     class_ = os.path.basename(filename)[:idx]
#     if not os.path.isdir(os.path.join(train_image_folder, class_)):
#         os.mkdir(os.path.join(train_image_folder, class_))
#     num_files = len(os.listdir(os.path.join(train_image_folder, class_)))
#     if num_files < 2000:
#         shutil.move(os.path.join(train_image_folder,filename), os.path.join(train_image_folder, class_))

# image_folder = pathlib.Path('data/nsynth-train/train_images copy')
# move_images_to_class_folders(image_folder)

##TO FLATTEN DIRECTORY
# dir_to_flatten = image_folder
# for dirpath, dirnames, filenames in os.walk(dir_to_flatten):
#     for filename in filenames:
#         try:
#             os.rename(os.path.join(dirpath, filename), os.path.join(dir_to_flatten, filename))
#         except OSError:
#             print ("Could not move %s " % os.path.join(dirpath, filename))


#from other flow from directory:


# #train set
# train_data_dir = pathlib.Path('data/nsynth-train/train_images')
# train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
#             # preprocessing_function=preprocess_input,
#             # horizontal_flip=True,
#             # width_shift_range=3,
#             # height_shift_range=3,
#             # brightness_range=None,
#             # shear_range=3,
#             # zoom_range=3,
#             # channel_shift_range=3,
#             rescale=1./255
#             )
# train_generator = train_gen.flow_from_directory(
#                     directory=train_data_dir,
#                     seed=52,
#                     color_mode='rgb', #just uncommented this
#                     target_size=(img_height, img_width),
#                     batch_size=batch_size,
#                     class_mode='sparse',
#                     shuffle=True
#                     )
# #valid set
# valid_data_dir = pathlib.Path('data/nsynth-valid/valid_images')
# valid_gen = tf.keras.preprocessing.image.ImageDataGenerator(
#             # preprocessing_function=preprocess_input,
#             rescale=1./255
#             )
# valid_generator = valid_gen.flow_from_directory(
#                     directory=valid_data_dir,
#                     seed=52,
#                     color_mode='rgb', #just uncommented this
#                     target_size=(img_height, img_width),
#                     batch_size=batch_size,
#                     class_mode='sparse',
#                     shuffle=True
#                 )


# #test set: using test images
# test_data_dir = pathlib.Path('data/nsynth-test/test_images')
# test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
#             # preprocessing_function=preprocess_input,
#             rescale=1./255
#             )
# test_generator = test_gen.flow_from_directory(
#                     directory=test_data_dir,
#                     seed=12,
#                     color_mode='rgb', #just uncommented this
#                     target_size=(img_height, img_width),
#                     batch_size=1,
#                     shuffle=False,
#                     class_mode='sparse'
#                     )




# #test set: using test images
# test_data_dir = pathlib.Path('data/nsynth-test/test_images')
# test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
#             rescale=1./255
#             )
# test_generator = test_gen.flow_from_directory(
#     directory=test_data_dir,
#     seed=12,
#     color_mode='rgb',
#     target_size=(img_height, img_width),
#     batch_size=1,
#     shuffle=False,
#     class_mode='sparse'
#     )

#cnn training
#trail 7: no extra dropout layer of .25, learning rate of .0001 ('7 og')
#*trial 8: (change plot names)- added dropout .25 layer, changed learning rate to .001 ('7')
#trial 9: keep learning rate .001, add another dropout layer
#trial 10 (lr: .001),
#trial 11 (lr: .0005), no stopping condition:
#trail 12: lr=.0005, no stopping, add batch normalization and dropout layers
#trial 12: added augmentation, went back to original model's drop out layers


#from transfer model

#train set
    # train_data_dir = pathlib.Path('data/nsynth-train/train_images')
    # train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    #             preprocessing_function=preprocess_input,
    #             # horizontal_flip=True,
    #             # width_shift_range=3,
    #             # height_shift_range=3,
    #             # brightness_range=None,
    #             # shear_range=3,
    #             # zoom_range=3,
    #             # channel_shift_range=3,
    #             # rescale=1./255
    #             )
    # train_generator = train_gen.flow_from_directory(
    #                   directory=train_data_dir,
    #                   seed=52,
    #                   color_mode='rgb', #just uncommented this
    #                   target_size=(img_height, img_width),
    #                   batch_size=batch_size,
    #                   class_mode='sparse',
    #                   shuffle=True
    #                   )
    # #valid set
    # valid_data_dir = pathlib.Path('data/nsynth-valid/valid_images')
    # valid_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    #             preprocessing_function=preprocess_input
    #             )
    # valid_generator = valid_gen.flow_from_directory(
    #                   directory=valid_data_dir,
    #                   seed=52,
    #                   color_mode='rgb', #just uncommented this
    #                   target_size=(img_height, img_width),
    #                   batch_size=batch_size,
    #                   class_mode='sparse',
    #                   shuffle=True
    #                 )

        # #test set: using test images
    # test_data_dir = pathlib.Path('data/nsynth-test/test_images')
    # test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    #             preprocessing_function=preprocess_input
    #             )
    # test_generator = test_gen.flow_from_directory(
    #                  directory=test_data_dir,
    #                  seed=12,
    #                  color_mode='rgb', #just uncommented this
    #                  target_size=(img_height, img_width),
    #                  batch_size=1,
    #                  shuffle=False,
    #                  class_mode='sparse'
    #                  )


    ###RESULTS/PLOTS-----------------------
    # last_unfreeze_layer = 76
    # #load model
    # transfer_model_filepath=f'saved_models/best_transfer_model_{last_unfreeze_layer}.hdf5'
    # transfer_model = tensorflow.keras.models.load_model(transfer_model_filepath)
    # print(transfer_model.summary())
    # test_loss, test_accuracy = transfer_model.evaluate(test_generator, verbose=2)
    # print(f"test_loss = {test_loss}, test_accuracy = {test_accuracy}")
    # y_probas = transfer_model.predict(test_generator)
    # y_preds = np.argmax(y_probas, axis=1)
    # y_actual = test_generator.classes

    #PLOT RESULTS
    #plot learning curves
    # fig, ax = plt.subplots(figsize=(8, 5))
    # plot_learning_curves(history, 'trial6', ax=ax)

    # # Compute confusion matrix
    # cm = confusion_matrix(y_actual, y_preds)
    # np.set_printoptions(precision=2)

    # # Plot confusion matrix
    # class_names = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet',
    #                 'organ', 'reed', 'string', 'synth_lead', 'vocal']
    # plot_confusion_matrix(cm, classes=class_names,
    #                       title='Confusion Matrix', image_name=f'{unfreeze_layer}')




    # ##INSTANTIATE TRANSFER MODEL
    # transfer_model = create_transfer_model(input_size=(100,100,3), n_categories=num_classes)
    # print_model_properties(transfer_model)
    # change_trainable_layers(transfer_model, 132)
    # print_model_properties(transfer_model)

    # #TRAIN MODEL ON NEW DATA, LAYER BY LAYER
    # #params
    # unfreeze_layer = 132
    # optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1 = 0.9, beta_2 = 0.999)
    # callbacks = [
    #     tf.keras.callbacks.EarlyStopping(monitor="val_loss",
    #                                   patience=10,
    #                                   verbose=1),
    #     tf.keras.callbacks.ModelCheckpoint(filepath=f'saved_models/best_transfer_model{unfreeze_layer}.hdf5',
    #                                     verbose=1, save_best_only=True, monitor='val_loss')
    #             ]
    # transfer_model.compile(optimizer=optimizer,
    #               loss=tf.losses.SparseCategoricalCrossentropy(),
    #               metrics=['accuracy'])

    # history = transfer_model.fit(
    #     train_generator,
    #     validation_data=valid_generator,
    #     epochs=50,
    #     verbose=2,
    #     callbacks=callbacks
    #     )

    # # # test_loss, test_accuracy = transfer_model.evaluate(test_generator)
    # # # print('Test accuracy :', test_accuracy)
    # #get metrics
    # list_of_axvline_values = []
    # acc = history.history['accuracy']
    # val_acc = history.history['accuracy']
    # loss = history.history['accuracy']
    # val_loss = history.history['accuracy']
    # plot_accuracy_loss(acc, val_acc, loss, val_loss, list_of_axvline_values, unfreeze_layer)

    # #pickle metrics and history (for next iteration)
    # save_pickle_files(acc, 'pickles/accuracy.pkl')
    # save_pickle_files(val_acc, 'pickles/val_accuracy.pkl')
    # save_pickle_files(loss, 'pickles/val_loss.pkl')
    # save_pickle_files(val_loss, 'pickles/val_loss.pkl')
    # save_pickle_files(history, 'pickles/history.pkl')
    # save_pickle_files(list_of_axvline_values, 'pickles/list_of_axvline_values.pkl')

    #FINE TUNE MODEL, LAYER BY LAYER



    # #MAYBE CAN BREAK THIS APART FROM INITIAL MODEL
    # last_unfreeze_layer=132 #update these w/ each iter
    # unfreeze_layer = 116 #update these w/ each iter
    # #get past metrics
    # acc = get_pickle_files('pickles/accuracy.pkl')
    # val_acc = get_pickle_files('pickles/val_accuracy.pkl')
    # loss = get_pickle_files('pickles/loss.pkl')
    # val_loss = get_pickle_files('pickles/val_loss.pkl')
    # last_history = get_pickle_files('pickles/last_history.pkl')
    # list_of_axvline_values = get_pickle_files('pickles/list_of_axvline_values.pkl')
    # intitial_epochs = last_history.epochs[-1]
    # list_of_axvline_values.append(initial_epochs)
    # #save for next iter
    # save_pickle_files(list_of_axvline_values, 'pickles/list_of_axvline_values.pkl')
    # #set up epochs to accumulate for plotting
    # fine_tune_epochs=50
    # total_epochs = initial_epochs + fine_tune_epochs
    # #open saved transfer model
    # transfer_model_filepath=f'saved_models/best_transfer_model{last_unfreeze_layer}.hdf5'
    # transfer_model = keras.models.load_model(transfer_model_filepath)
    # print(transfer_model.summary())
    # change_trainable_layers(transfer_model, unfreeze_layer) #update trainable layers, block by block
    # print_model_properties(transfer_model)

    # transfer_model.compile(optimizer=optimizer,
    #                        loss=tf.losses.SparseCategoricalCrossentropy(),
    #                        metrics=['accuracy'])

    # history = transfer_model.fit(
    #                     train_generator,
    #                     validation_data=valid_generator,
    #                     epochs=total_epochs,
    #                     initial_epoch = initial_epochs,
    #                     verbose=2,
    #                     callbacks=callbacks
    #                     )

    # #update metrics now + pickle updated version (for next iteration)
    # acc = update_metrics('accuracy', history)
    # val_acc = update_metrics('val_accuracy', history)
    # loss = update_metrics('loss', history)
    # val_loss = update_metrics('val_loss', history)
    # #save history as pickle
    # save_pickle_files(history, 'pickles/history.pkl')
    # # plot
    # plot_accuracy_loss(acc, val_acc, loss, val_loss, list_of_axvline_values, unfreeze_layer)






    ###RESULTS/PLOTS-----------------------
    # # test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    # test_loss, test_accuracy = transfer_model.evaluate(test_generator, verbose=2)
    # print(f"test_loss = {test_loss}, test_accuracy = {test_accuracy}")
    # y_probas = transfer_model.predict(test_generator)
    # y_preds = np.argmax(y_probas, axis=1)
    # y_actual = test_generator.classes

    # #PLOT RESULTS
    # #plot learning curves
    # fig, ax = plt.subplots(figsize=(8, 5))
    # plot_learning_curves(history, 'trial6', ax=ax)

    # # Compute confusion matrix
    # cm = confusion_matrix(y_actual, y_preds)
    # np.set_printoptions(precision=2)

    # # Plot confusion matrix
    # class_names = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet',
    #                 'organ', 'reed', 'string', 'synth_lead', 'vocal']
    # plot_confusion_matrix(cm, classes=class_names,
    #                       title='Confusion Matrix', image_name='trial6')



 # #added to unfreeze batchnormalized layers
    # for layer in transfer_model.layers:
    #     # layer.trainable = False
    #     # if isinstance(layer, tf.keras.layers.BatchNormalization):
    #     #     layer._per_input_updates = {}
    #     # if "batch_normalization" in layer.__class__.__name__:
    #     #     layer.trainable = True
    #     if isinstance(layer, tf.keras.layers.BatchNormalization):
    #         layer.trainable = True








    #   # #feature importances
#     feat_scores_array = model.feature_importances_
#     feat_scores_df = pd.DataFrame({'Fraction of Samples Affected by Feature' : model.feature_importances_},
#                             )
#     feat_scores = feat_scores_df.sort_values(by='Fraction of Samples Affected by Feature', ascending=False)

    # # #plot top 40 features
    # fig, ax = plt.subplots()
    # x_pos = np.arange(len(feat_scores[:40]))
    # ax.barh(x_pos, feat_scores['Fraction of Samples Affected by Feature'][:40], align='center')
    # plt.yticks(x_pos, feat_scores.index[:40])
    # ax.set_ylabel('MFCC Coefficients')
    # ax.set_xlabel('Fraction of Samples Affected')
    # ax.set_title('MFCC Coefficients by Importance')
    # plt.gca().invert_yaxis()
    # plt.savefig('images/top40_rf_feature_importances.png', bbox_inches = "tight")
    # plt.show()

    # # #plot top 15 features
    # fig, ax = plt.subplots()
    # x_pos = np.arange(len(feat_scores[:15]))
    # ax.barh(x_pos, feat_scores['Fraction of Samples Affected by Feature'][:15], align='center')
    # plt.yticks(x_pos, feat_scores.index[:15])
    # ax.set_ylabel('MFCC Coefficients')
    # ax.set_xlabel('Fraction of Samples Affected')
    # ax.set_title('Top 15 MFCC Coefficients by Importance')
    # plt.gca().invert_yaxis()
    # plt.savefig('images/top20_rf_feature_importances.png', bbox_inches = "tight")
    # plt.show()
