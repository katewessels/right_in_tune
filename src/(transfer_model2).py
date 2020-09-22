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
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling1D, GlobalAveragePooling2D, AveragePooling2D, Input, Add
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
from keras.applications.xception import preprocess_input
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
#2nd iteration:
# got rid fo preprocessing augmentation

#GET DATA GENERATORS
#params
batch_size = 32
img_height = 150
img_width = 150
num_classes = 11
#train set
train_data_dir = pathlib.Path('data/nsynth-train/train_images')
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=preprocess_input,
            # horizontal_flip=True,
            # width_shift_range=3,
            # height_shift_range=3,
            # brightness_range=None,
            # shear_range=3,
            # zoom_range=3,
            # channel_shift_range=3,
            # rescale=1./255
            )

train_generator = train_gen.flow_from_directory(
    directory=train_data_dir,
    seed=52,
    # color_mode='rgb',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse',
    shuffle=True
    )
#valid set
valid_data_dir = pathlib.Path('data/nsynth-valid/valid_images')
valid_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=preprocess_input
            )
valid_generator = valid_gen.flow_from_directory(
    directory=valid_data_dir,
    seed=52,
    # color_mode='rgb',
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
    # color_mode='rgb',
    target_size=(img_height, img_width),
    batch_size=1,
    shuffle=False,
    class_mode='sparse'
    )

#CREATE BASE MODEL
def create_transfer_model(input_size, n_categories, weights = 'imagenet'):
        base_model = Xception(weights=weights,
                          include_top=False,
                          input_shape=input_size)

        model = base_model.output
        model = GlobalAveragePooling2D()(model)
        predictions = Dense(n_categories, activation='softmax')(model)
        model = Model(inputs=base_model.input, outputs=predictions)

        return model
# transfer_model = create_transfer_model(input_size=(100,100,3), n_categories=num_classes)


#display layers
def print_model_properties(model, indices = 0):
     for i, layer in enumerate(model.layers[indices:]):
        print("Layer {} | Name: {} | Trainable: {}".format(i+indices, layer.name, layer.trainable))

# print_model_properties(transfer_model)

#change trainable layers
def change_trainable_layers(model, trainable_index):
    for layer in model.layers[:trainable_index]:
        layer.trainable = False
    for layer in model.layers[trainable_index:]:
        layer.trainable = True

# change_trainable_layers(transfer_model, 132)
# print_model_properties(transfer_model)


#train the model on new data
optimizer = tf.keras.optimizers.Adam(lr=0.00001, beta_1 = 0.9, beta_2 = 0.999)

# transfer_model.compile(optimizer=optimizer,
#               loss=tf.losses.SparseCategoricalCrossentropy(), #from_logits=True
#               metrics=['accuracy'])

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        # min_delta=1e-2,
        patience=3,
        verbose=1),
    keras.callbacks.ModelCheckpoint(filepath='saved_models/best_transfer_model.hdf5',
                               verbose=1, save_best_only=True, monitor='val_loss')
            ]

# history = transfer_model.fit(
#     train_generator,
#     validation_data=valid_generator,
#     epochs=50,
#     verbose=2,
#     callbacks=callbacks
#     )

# ##PLOT
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# plt.figure(figsize=(8, 8))
# plt.subplot(2, 1, 1)
# plt.plot(acc, label='Training Accuracy')
# plt.plot(val_acc, label='Validation Accuracy')
# plt.legend(fontsize=14)
# plt.ylabel('Accuracy', fontsize=14)
# plt.ylim([min(plt.ylim()),1])
# plt.title('Training and Validation Accuracy', fontsize=14)

# plt.subplot(2, 1, 2)
# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.legend(fontsize=14)
# plt.ylabel('Cross Entropy', fontsize=14)
# plt.ylim([0,1.0])
# plt.title('Training and Validation Loss', fontsize=14)
# plt.xlabel('epoch', fontsize=14)
# plt.savefig('images/transfer_layer1.png', bbox_inches='tight')
# plt.show()

# loss, accuracy = transfer_model.evaluate(test_generator)
# print('Test accuracy :', accuracy)

#NEXT ITERATION
#unfreeze through next block
#updatefile name for saved plot
#add vertical line for each phase
#get values from last iteration: acc, val_acc, loss, val_loss, last epoch
#from model1:
acc = [0.2154376357793808,
 0.25472089648246765,
 0.2710314691066742,
 0.28417184948921204,
 0.28614747524261475,
 0.29418790340423584,
 0.2939581871032715,
 #
 0.2604181170463562,
 0.3167470693588257,
 0.3408224284648895,
 0.35768434405326843,
 0.3728463053703308,
 0.3791867792606354,
 0.39416494965553284,
 #
 0.3162876069545746,
 0.3692166209220886,
 0.39535951614379883,
 0.4200780987739563,
 #
 0.39586493372917175,
 0.42701584100723267,
 0.4418562054634094,
 0.4526073932647705]
val_acc = [0.20405426621437073,
 0.23994320631027222,
 0.18796339631080627,
 0.2591891586780548,
 0.24491244554519653,
 0.2445969432592392,
 0.28230005502700806,
 #
 0.28876793384552,
 0.2375769019126892,
 0.28340432047843933,
 0.2755955159664154,
 0.2486196607351303,
 0.2602934241294861,
 0.18693800270557404,
 #
 0.21099542081356049,
 0.2520902454853058,
 0.2993374466896057,
 0.3058053255081177,
 #
 0.3237103521823883,
 0.33041489124298096,
 0.33270230889320374,
 0.33790817856788635]
loss = [2.203662157058716,
 2.11171555519104,
 2.074953079223633,
 2.0463008880615234,
 2.0285868644714355,
 2.0159244537353516,
 2.0076541900634766,
 #
 2.0713047981262207,
 1.917178750038147,
 1.8470869064331055,
 1.7966725826263428,
 1.7620489597320557,
 1.7303568124771118,
 1.7016173601150513,
 #
 1.9132083654403687,
 1.763192892074585,
 1.6894652843475342,
 1.6325820684432983,
 #
 1.69840407371521,
 1.6118323802947998,
 1.5638647079467773,
 1.5363041162490845]
val_loss = [2.1959736347198486,
 2.1341605186462402,
 2.198939085006714,
 2.040144205093384,
 2.1343557834625244,
 2.106779098510742,
 2.1925268173217773,
 #
 2.4206008911132812,
 2.689269781112671,
 2.325876474380493,
 2.5452282428741455,
 2.6631412506103516,
 2.7958061695098877,
 3.109011650085449,
 #
 2.293417453765869,
 2.6679089069366455,
 2.593614339828491,
 2.6048102378845215,
 #
 2.50100040435791,
 2.6394243240356445,
 2.7949326038360596,
 2.9080214500427246]

initial_epochs = 18
#history.epoch[-1] = 6, 12, 15, 18

fine_tune_epochs=50
total_epochs = initial_epochs + fine_tune_epochs


#open saved transfer model
transfer_model_filepath='saved_models/best_transfer_model.hdf5'
transfer_model = keras.models.load_model(transfer_model_filepath)
print(transfer_model.summary())
change_trainable_layers(transfer_model, 96) #update trainable layers, block by block
print_model_properties(transfer_model)

transfer_model.compile(optimizer=optimizer,
              loss=tf.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])


history_fine_tune = transfer_model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=total_epochs,
    initial_epoch = initial_epochs,
    verbose=2,
    callbacks=callbacks
    )

#keep checking to see if val_accuracy improves. if not, new model
#won't save, and can try again by adjusting learning rate, etc.


# accumulate metrics (along w/ epochs) for plotting
acc += history_fine_tune.history['accuracy']
val_acc += history_fine_tune.history['val_accuracy']
loss += history_fine_tune.history['loss']
val_loss += history_fine_tune.history['val_loss']


#plot
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
# plt.ylim([0, 1])
plt.ylabel('Accuracy', fontsize=14)
#fine tuning marks for each iteration of a deeper block in network
plt.axvline(6-1, color='green', label='Fine Tuning Iterations')
plt.axvline(12-1, color='green')
plt.axvline(15-1, color='green')
plt.axvline(18-1, color='green')
# plt.plot([initial_epochs-1,initial_epochs-1],
#           plt.ylim(), color='green', label='Fine Tuning')
plt.legend(fontsize=14)
plt.title('Training and Validation Accuracy', fontsize=14)

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
# plt.ylim([0, 1])
plt.ylabel('Cross Entropy', fontsize=14)
#fine tuning marks for each iteration of a deeper block in network
plt.axvline(6-1, color='green', label='Fine Tuning Iterations')
plt.axvline(12-1, color='green')
plt.axvline(15-1, color='green')
plt.axvline(18-1, color='green')
# plt.plot([6,6],plt.ylim(), color='green', label='Fine Tuning')
plt.legend(fontsize=14)
plt.title('Training and Validation Loss', fontsize=14)
plt.xlabel('epoch', fontsize=14)
plt.savefig('images/transfer_layer5.png', bbox_inches='tight')
plt.show()


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
