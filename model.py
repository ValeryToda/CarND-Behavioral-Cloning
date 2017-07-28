import tensorflow as tf
from keras.models import Sequential
import keras.backend as K
from keras.layers import Dense, Conv2D, Flatten, Lambda, Cropping2D, Dropout, BatchNormalization, MaxPool2D, Activation
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.callbacks import LearningRateScheduler
from keras import initializers
from keras import regularizers
from keras import optimizers
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
from collections import namedtuple
import matplotlib.pyplot as plt
import csv
import cv2
import pickle
import numpy as np
import math
from data import balance_data, generator, is_gray_mode
from utils import threadsafe_iter

train_generator = None
validation_generator = None

def plot_loss(history_object):
    plt.figure(num=30)
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

def save_model_topology(model):
    # serialize keras model into JSON file
    model_json = model.to_json()
    with open("./_cnnet/model_topology.json", "w") as json_file:
        json_file.write(model_json)
    print("Model saved.")

def load_model(json_filepath, weights_filepath, save_binary=False):
    # load weights into a model and 
    # save binary version 
    
    # load json and create model
    json_file = open(json_filepath, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights_filepath)
    #./_weights/w-10-0.20.hdf5
    if save_binary:
        loaded_model.save('./_cnnet/model.h5')
    return loaded_model


def nvidia_model(in_shape):
    ## model derived from nvidia model architecture
    # https://arxiv.org/pdf/1604.07316v1.pdf
    # Dropouts and RELUs added

    model = Sequential()
    model.add(Conv2D(24, (5, 5), input_shape=in_shape, name='cv_1'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), padding='same', name='cv_2'))
    
    model.add(BatchNormalization(name='bn_00'))    
    model.add(Dropout(0.5, name='do_00'))    
    model.add(Conv2D(48, (5, 5), strides=(2, 2), padding='same', name='cv_3'))

    model.add(BatchNormalization(name='bn_01'))    
    model.add(Dropout(0.5, name='do_01'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='cv_4'))

    model.add(BatchNormalization(name='bn_0'))    
    model.add(Dropout(0.5, name='do_0'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='cv_5'))
    
    model.add(Flatten(name='ft_1'))

    model.add(BatchNormalization(name='bn_0_0'))    
    model.add(Dropout(0.5, name='do_0_0'))
    model.add(Dense(100, name='fc_1'))
    model.add(Activation('relu'))

    model.add(BatchNormalization(name='bn_1'))
    model.add(Dropout(0.25, name='do_1'))
    model.add(Dense(50, name='fc_2'))
    model.add(Activation('relu'))
    
    model.add(BatchNormalization(name='bn_2'))
    model.add(Dropout(0.1, name='do_2'))
    model.add(Dense(10, name='fc_3'))
    model.add(Activation('relu'))

    model.add(Dense(1, name='out'))
    return model


def train(hyper_params, ropl=True, resume=[]):
    """
    Training routine
    @param hyper_params: namedtuple with hyper params configuration
    @param gray: choose between training with colored or grayscaled images
    """
    K.clear_session()

    if len(resume) > 0:
        model = load_model(resume[0], resume[1])
    else:
        model = nvidia_model(hyper_params.in_shape)

    # for SGD optimizer
    model.compile(optimizer=optimizers.Adam(lr=hyper_params.learning_rate), loss='mean_squared_error', metrics=['accuracy'])
    ## Model Summary
    model.summary()
    ## Save the model topology
    save_model_topology(model)
    ## Checkpoint
    filepath="./_weights/w-{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpointer = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,\
                                    save_best_only=True)

    # reduce the learnrate on plateau during training 
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.1e-6, verbose=1)
    
    # Callback list. will be call during training 
    callbacks_list = [checkpointer]
    if ropl:
        callbacks_list.append(reduce_lr)

    hist = None
    hist_file = "./_cnnet/history.p"
    _saved = False
    aborted_by_user = False

    try:
        hist = model.fit_generator(train_generator,\
                    steps_per_epoch = hyper_params.steps_per_epoch,\
                    validation_data = validation_generator,\
                    validation_steps = hyper_params.nb_val_samples,\
                    verbose=1,\
                    epochs=hyper_params.epochs,\
                    callbacks=callbacks_list, workers=5)
        
    except KeyboardInterrupt:
        print("Training aborted by user")    
        model.save('./_cnnet/model.h5')
        print("Model saved.")
        _saved = True
        aborted_by_user = True
        
    if not aborted_by_user:
        ## Save history to file
        out_tr_file = open(hist_file, 'wb')
        hist_container = {'acc': hist.history['acc'], 'val_acc': hist.history['val_acc'],\
                      'loss': hist.history['loss'], 'val_loss': hist.history['val_loss']}
        pickle.dump(hist_container, out_tr_file)
        out_tr_file.close()
    
    if not _saved:
        print("Training history saved.")
        model.save('./_cnnet/model.h5')
        print("Model saved.")


if __name__ == '__main__':
    hyper_params = namedtuple('HyperParameters', 'epochs batch learning_rate\
                             nb_val_samples optimizer samples_per_epoch in_shape')

    ## load data database from exported files 
    samples = balance_data(["./_data/export_2017_07_23_12_23_46.csv",
                            "./_data/export_2017_07_23_12_25_33.csv",
                            "./_data/export_2017_07_23_12_25_43.csv"], False, reduce_factor=0.00)

    # shuffle the dataframe
    samples = shuffle(samples)

    # split for training (90%) and test (10%)
    train_samples, validation_samples = train_test_split(samples, test_size=0.1)

    # declare generators
    hyper_params.batch = 256
    tr_generator = generator(train_samples, batch_size=hyper_params.batch)
    train_generator = tr_generator
    
    val_generator = generator(validation_samples, batch_size=hyper_params.batch)
    validation_generator = val_generator
    
    # set hyperparams
    hyper_params.epochs = 10
    hyper_params.steps_per_epoch = len(train_samples)
    hyper_params.nb_val_samples = len(validation_samples)    
    hyper_params.learning_rate = 1e-4

    # Set image input size
    hyper_params.in_shape = (16, 64, 3)
    if is_gray_mode():
        hyper_params.in_shape = (16, 64, 1)

    # Call training routine!
    train(hyper_params, ropl=False)


