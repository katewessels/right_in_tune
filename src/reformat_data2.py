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
# import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization, Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling1D, AveragePooling2D, Input, Add
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from datetime import datetime
from dumb_cnn import get_pickle_files, get_X_y
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix #, plot_confusion_matrix
import itertools
from plot_models import plot_learning_curves, plot_confusion_matrix
import seaborn as sns
sns.set()
import tensorflow as tf
from tensorflow.keras import backend as k
from tensorflow.python.keras import backend as k


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

# batch_size = 32
batch_size= 16
img_height = 64
img_width = 64


#train set
train_data_dir = pathlib.Path('data/nsynth-train/train_images')
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            # preprocessing_function=preprocess_input,
            # horizontal_flip=True,
            # width_shift_range=3,
            # height_shift_range=3,
            # brightness_range=None,
            # shear_range=3,
            # zoom_range=3,
            # channel_shift_range=3,
            rescale=1./255
            )
train_generator = train_gen.flow_from_directory(
                    directory=train_data_dir,
                    seed=52,
                    color_mode='rgb', #just uncommented this
                    target_size=(img_height, img_width),
                    batch_size=batch_size,
                    class_mode='sparse',
                    shuffle=True
                    )
#valid set
valid_data_dir = pathlib.Path('data/nsynth-valid/valid_images')
valid_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            # preprocessing_function=preprocess_input,
            rescale=1./255
            )
valid_generator = valid_gen.flow_from_directory(
                    directory=valid_data_dir,
                    seed=52,
                    color_mode='rgb', #just uncommented this
                    target_size=(img_height, img_width),
                    batch_size=batch_size,
                    class_mode='sparse',
                    shuffle=True
                )
#test set: using test images
test_data_dir = pathlib.Path('data/nsynth-test/test_images')
test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            # preprocessing_function=preprocess_input,
            rescale=1./255
            )
test_generator = test_gen.flow_from_directory(
                    directory=test_data_dir,
                    seed=12,
                    color_mode='rgb', #just uncommented this
                    target_size=(img_height, img_width),
                    batch_size=1,
                    shuffle=False,
                    class_mode='sparse'
                    )
# #train set
# train_data_dir = pathlib.Path('data/nsynth-train/train_images')
# train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
#             rescale=1./255,
#             validation_split=0.2
#             )
# train_generator = train_gen.flow_from_directory(
#     directory=train_data_dir,
#     seed=52,
#     subset='training',
#     color_mode='rgb',
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='sparse',
#     shuffle=True
#     )

# #valid set
# valid_generator = train_gen.flow_from_directory(
#     directory=train_data_dir,
#     seed=52,
#     subset='validation',
#     color_mode='rgb',
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='sparse',
#     shuffle=True
#     )

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


##TRAIN MODEL
#trail 7: no extra dropout layer of .25, learning rate of .0001 ('7 og')
#*trial 8: (change plot names)- added dropout .25 layer, changed learning rate to .001 ('7')
#trial 9: keep learning rate .001, add another dropout layer
num_classes = 11
num_channels = 3

#trial 10 (lr: .001),
#trial 11 (lr: .0005), no stopping condition:
#trail 12: lr=.0005, no stopping, add batch normalization and dropout layers
#trial 12: added augmentation, went back to original model's drop out layers
#ORIGINAL:
# model = Sequential()
# model.add(Conv2D(32, (3, 3), padding='same',
#                 input_shape=(img_height, img_width, num_channels)))
# model.add(Activation('relu'))
# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# # model.add(BatchNormalization()) #added
# # model.add(Dropout(0.1)) #added
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Conv2D(64, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# # model.add(BatchNormalization()) #added
# # model.add(Dropout(0.2)) #added
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))
# model.add(Conv2D(128, (3, 3), padding='same'))
# model.add(Activation('relu'))
# # model.add(Dropout(0.5)) #added
# model.add(Conv2D(128, (3, 3)))
# model.add(Activation('relu'))
# # model.add(BatchNormalization()) #added
# # model.add(Dropout(0.5)) #added
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))
# model.add(Flatten())
# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))

#mods
# model = Sequential()
# model.add(Conv2D(32, (3, 3), padding='same',
#                 input_shape=(img_height, img_width, num_channels)))
# model.add(Activation('relu'))
# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(BatchNormalization()) #added
# model.add(Dropout(0.5)) #added
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Conv2D(64, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(BatchNormalization(momentum=.99)) #added
# model.add(Dropout(0.5)) #added
# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(BatchNormalization(momentum=.99)) #added
# model.add(Dropout(0.5)) #added
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))
# model.add(Conv2D(128, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(BatchNormalization(momentum=.99)) #added
# model.add(Dropout(0.5)) #added
# model.add(Conv2D(128, (3, 3)))
# model.add(Activation('relu'))
# model.add(BatchNormalization(momentum=.99)) #added
# # model.add(Dropout(0.5)) #added
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))
# model.add(Flatten())
# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))

#just a try:
#Building model 1 using customized convolutional and pooling layers

model = Sequential()
model.add(Conv2D(filters = 16, kernel_size = 2,input_shape=(100,100,3),padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization(momentum=.99)) #added
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters = 32,kernel_size = 2,activation= 'relu',padding='same'))
model.add(BatchNormalization(momentum=.99)) #added
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters = 64,kernel_size = 2,activation= 'relu',padding='same'))
model.add(BatchNormalization(momentum=.99)) #added
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters = 128,kernel_size = 2,activation= 'relu',padding='same'))
model.add(BatchNormalization(momentum=.99)) #added
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(150))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(11, activation='softmax'))
model.summary()

#lr was 0.0005
optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1 = 0.9, beta_2 = 0.999)

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


# for using image generators, flow from dir
history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=150,
    verbose=2 #,
    # callbacks=callbacks  #for trial11 got rid of early stopping
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
saved_model_path = "./saved_models/cnn_with_train11{}.h5".format(datetime.now().strftime("%m%d")) # _%H%M%S
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
plt.savefig('images/accuracy_loss11.png', bbox_inches='tight')
plt.show()


#PLOT RESULTS
#plot learning curves
fig, ax = plt.subplots(figsize=(8, 5))
plot_learning_curves(history, 'trial11', ax=ax)

# Compute confusion matrix
cm = confusion_matrix(y_actual, y_preds)
np.set_printoptions(precision=2)

# Plot confusion matrix
class_names = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet',
                'organ', 'reed', 'string', 'synth_lead', 'vocal']
plot_confusion_matrix(cm, classes=class_names,
                      title='Confusion Matrix', image_name='trial11')
