#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 20:52:20 2019

@author: austin
"""
import os
import tensorflow as tf

PROJ_DIR = os.environ['PROJ_DIR']
MODEL_DIR = PROJ_DIR + '/Classification/models'

def get_model(w, h, c, n_classes):
    '''Model for the spatial image'''
    # This returns a tensor
    inputs = tf.keras.layers.Input(shape=(w, h, c), name='spatial_img')
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(10, activation='relu', name='spatial_HL1')(x)
    x = tf.keras.layers.Dense(10, activation='relu', name='spatial_HL2')(x)
    x = tf.keras.layers.Dense(10, activation='relu', name='spatial_HL3')(x)
    x = tf.keras.layers.Dense(10, activation='relu', name='spatial_HL4')(x)
    outputs = tf.keras.layers.Dense(n_classes, activation='relu', name='spatial_output')(x)
    model= tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model

def compile_model(model):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
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