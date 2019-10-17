#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 20:52:20 2019

@author: austin
"""
import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout

PROJ_DIR = os.environ['PROJ_DIR']
MODEL_DIR = PROJ_DIR + '/Classification/models'

def get_model(w, h, c, n_classes, hl=3, max_pool=False, nnodes=200):
    '''Model for the spatial image'''
    # This returns a tensor
    inputs = tf.keras.layers.Input(shape=(w, h, c), name='spatial_img')
    x = inputs
    if max_pool:
        x = tf.keras.layers.MaxPooling2D(data_format='channels_last')(x)
    x = tf.keras.layers.Flatten()(x)
    for layer_num in range(hl):
        x = tf.keras.layers.Dense(220, activation='relu', name=f'spatial_HL{layer_num}')(x)
        #x = tf.keras.layers.Dropout(.35)(x)
    #x = tf.keras.layers.Dense(60, activation='relu', name='spatial_HL2')(x)
    #x = tf.keras.layers.Dense(60, activation='relu', name='spatial_HL3')(x)
    #x = tf.keras.layers.Dense(60, activation='relu', name='spatial_HL4')(x)
    #x = tf.keras.layers.Dense(60, activation='relu', name='spatial_HL5')(x)
    #x = tf.keras.layers.Dense(60, activation='relu', name='spatial_HL6')(x)
    #x = tf.keras.layers.Dense(60, activation='relu', name='spatial_HL7')(x)
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax', name='spatial_output')(x)
    model= tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model

def get_vgg_model(w, h, c, n_classes, hl=3, max_pool=False, nnodes=200):
    '''Model for the spatial image'''
    # This returns a tensor
    inputs = tf.keras.layers.Input(shape=(w, h, c), name='spatial_img')
    pad1_1 = ZeroPadding2D(padding=(1, 1))(inputs)
    conv1_1 = tf.keras.layers.Conv2D(16, 3, activation='relu', name='conv1_1')(pad1_1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2))(conv1_1)
    pad1_2 = ZeroPadding2D(padding=(1, 1))(pool1)
    conv1_2 = tf.keras.layers.Conv2D(16, 3, activation='relu', name='conv1_2')(pad1_2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2))(conv1_2)

    flat = Flatten()(pool2)
    fc6 = Dense(256, activation='relu', name='fc6')(flat)
    fc6_drop = Dropout(0.5)(fc6)
    fc7 = Dense(256, activation='relu', name='fc7')(fc6_drop)
    fc7_drop = Dropout(0.5)(fc7)
    outputs = Dense(256, activation='softmax', name='fc8')(fc7_drop)

    model= tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model

def get_vgg_model2(w, h, c, n_classes, hl=3, max_pool=False, nnodes=200):
    '''Model for the spatial image based on models proposed by
    VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION
    Karen Simonyan & Andrew Zisserman
    '''
    # This returns a tensor
    inputs = tf.keras.layers.Input(shape=(w, h, c), name='spatial_img')
    pad1 = ZeroPadding2D(padding=(1, 1))(inputs)
    conv1_1 = tf.keras.layers.Conv2D(16, 3, activation='relu', name='conv1_1')(pad1)
    conv1_2 = tf.keras.layers.Conv2D(16, 3, activation='relu', name='conv1_2')(conv1_1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2))(conv1_2)

    pad2 = ZeroPadding2D(padding=(1, 1))(pool1)
    conv2_1 = tf.keras.layers.Conv2D(32, 3, activation='relu', name='conv2_1')(pad2)
    conv2_2 = tf.keras.layers.Conv2D(32, 3, activation='relu', name='conv2_2')(conv2_1)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2_2)

    pad3 = ZeroPadding2D(padding=(1, 1))(pool2)
    conv3_1 = tf.keras.layers.Conv2D(64, 3, activation='relu', name='conv3_1')(pad3)
    conv3_2 = tf.keras.layers.Conv2D(64, 3, activation='relu', name='conv3_2')(conv3_1)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3_2)

    flat = Flatten()(pool3)
    fc1 = Dense(4096, activation='relu', name='fc1')(flat)
    fc1_drop = Dropout(0.5)(fc1)
    fc2 = Dense(4096, activation='relu', name='fc2')(fc1_drop)
    fc2_drop = Dropout(0.5)(fc2)
    fc3 = Dense(512, activation='relu', name='fc3')(fc2_drop)
    fc3_drop = Dropout(0.5)(fc3)
    outputs = Dense(256, activation='softmax', name='output')(fc3_drop)

    model= tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model

def compile_model(model):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
    return model


def save_trained_model(model, model_name):
    'Saves a model in the models folder and returns its path'
    model_path = MODEL_DIR + '/' + model_name + '.h5'
    print('Saving Model:', model_path)
    model.save(model_path)
    return model_path

def load_trained_model(model_name):
    'Load a pre-trained model'
    model_path = MODEL_DIR + '/' + model_name + '.h5'
    print('Loading Model:', model_path)
    if os.path.isfile(model_path):
        model = tf.keras.models.load_model(model_path)
        return model
    else:
        print('Not a valid model!')
        return None

def visualize_model(model):
    model.summary()