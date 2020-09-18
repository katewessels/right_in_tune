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

def move_images_to_class_folders(image_folder, file_limit=False):
    for filename in os.listdir(image_folder):
        idx = filename.find('_')
        class_ = os.path.basename(filename)[:idx]
        if not os.path.isdir(os.path.join(image_folder, class_)):
            os.mkdir(os.path.join(image_folder, class_))
        if file_limit==False:
            shutil.move(os.path.join(image_folder,filename), os.path.join(image_folder, class_))
        else:
            num_files = len(os.listdir(os.path.join(image_folder, class_)))
            if num_files < 2000:
                shutil.move(os.path.join(image_folder,filename), os.path.join(image_folder, class_))



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

###TO FLATTEN DIRECTORY
# dir_to_flatten = train_image_folder
# for dirpath, dirnames, filenames in os.walk(dir_to_flatten):
#     for filename in filenames:
#         try:
#             os.rename(os.path.join(dirpath, filename), os.path.join(dir_to_flatten, filename))
#         except OSError:
#             print ("Could not move %s " % os.path.join(dirpath, filename))



# # ##GET DATASETS (MATRICES)
# data_dir = pathlib.Path('data/nsynth-train/train_images')
batch_size = 64
img_height = 180
img_width = 180

# # #use validation set as training set for now
# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     data_dir,
#     validation_split=0.2,
#     subset='training',
#     seed=123,
#     color_mode='rgba',
#     image_size=(img_height, img_width),
#     batch_size=batch_size)

# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#   data_dir,
#   validation_split=0.2,
#   subset="validation",
#   seed=123,
#   color_mode='rgba',
#   image_size=(img_height, img_width),
#   batch_size=batch_size)

# #test set
# test_data_dir = pathlib.Path('data/nsynth-test/test_images')
# test_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     test_data_dir,
#     seed=1234,
#     color_mode='rgba',
#     image_size=(img_height, img_width),
#     batch_size=batch_size,
#     shuffle=False
#     )

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

# #test set (using validation images)
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
    batch_size=batch_size,
    shuffle=False,
    class_mode='sparse'
    )


#CONFIGURE DATASET FOR PERFORMANCE
#cache: keeps images in memory after they're loaded
#off disk during first epoch
#prefetch: overlaps data preprocessing and model execution while
#training
# AUTOTUNE = tf.data.experimental.AUTOTUNE
# train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

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
        keras.layers.Dropout(0.5),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation='softmax')
        ])

optimizer = tf.keras.optimizers.Adam(lr=0.0001, beta_1 = 0.9, beta_2 = 0.999)

model.compile(optimizer=optimizer,
              loss=tf.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])


callbacks = [
    keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor="val_loss",
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-2,
        # "no longer improving" being further defined as "for at least 2 epochs"
        patience=3,
        verbose=1,
    )]

# #for using datasets from dir
# history = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=10,
#     verbose=2
#     )

# for using image generators, flow from dir
history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=100,
    verbose=2,
    callbacks=callbacks
    )


print(model.summary())

# # #EVALUATE MODEL ON TEST SET

# test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
test_loss, test_accuracy = model.evaluate(test_generator, verbose=2)
print(f"test_loss = {test_loss}, test_accuracy = {test_accuracy}")
y_probas = model.predict(test_generator)
y_preds = np.argmax(y_probas, axis=1)
y_actual = test_generator.classes


#save model to an HDF5 file
saved_model_path = "./saved_models/cnn_with_train4{}.h5".format(datetime.now().strftime("%m%d")) # _%H%M%S
model.save(saved_model_path)

# #load/open a saved model
# saved_model_path = "./saved_models/cnn20917.h5"
# new_model = keras.models.load_model(saved_model_path)
# print(new_model.summary())

#PLOTS
# Plotting
plt.figure(figsize=(15,7))
#plot 1: accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
#plot 2:loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('images/accuracy_loss.png', bbox_inches='tight')
plt.show()



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cm = confusion_matrix(y_actual, y_preds)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
class_names = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet',
                'organ', 'reed', 'string', 'synth_lead', 'vocal']
plt.figure()
plot_confusion_matrix(cm, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
plt.savefig('images/confusion_matrix.png', bbox_inches='tight')
plt.show()
