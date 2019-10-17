'''
Driver

@author austin
'''

import data
import os
from logger import Logger
import numpy as np
import tensorflow as tf
import models
import time

version = '1.0.0'

change_history = '''
date       author    version    message
###############################################################
09/25/19   austin    1.0.0      Initial version
###############################################################
'''


def about():
    print('-------------------------------------------------')
    print('Name:', os.path.basename(__file__))
    print('            Change History',change_history)
    print('Current Version:', version)
    print('-------------------------------------------------')


#data.about()

logger = Logger(debug=1)

height = 128
width = 128
channels = 1
images = data.load_jaffe_images(height, width, channels, face=True, edge=False)


logger.put_msg("I", f"Processing {len(images)} images")


x = lambda a: [a.subject_id, a.emotion_class, a.img_array]

def lcd(num, d_num=2):
    d_count = 0
    for i in range(3, 25):
        if num % i == 0:
            d_count += 1
            if d_count == d_num:
                return i
    print('No lcd')
    exit()

subjects = map(x, images)
data = np.array(list(subjects))
extra_data = data[0:3]
data = data[3:]
num_subjects = len(data)
print(f'{num_subjects} images')
validation_size = lcd(num_subjects, d_num=8)
logger.put_msg("I", f"Validation Set Size: {validation_size}")
num_iterations = int(num_subjects / validation_size)


start = 0
end = 0
results = []
start_time = time.time()
for idx in range(num_iterations):
    iteration_start_time = time.time()
    print(f'Starting iteration {idx} of {num_iterations}')
    start = idx * validation_size
    end = start + validation_size
    if extra_data is not None:
        test = np.concatenate((data[start:end, :], extra_data), axis=0)
    else:
        test = data[start:end, :]

    x_test = test[:,2]
    x_test = np.stack(x_test)

    y_test = test[:,1]
    y_test = np.stack(y_test)

    train = np.concatenate((data[0:start,:], data[end:,:]), axis=0)
    x_train = train[:,2]
    x_train = np.stack(x_train)

    y_train = train[:,1]
    y_train = np.stack(y_train)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    n_classes = len(np.unique(y_train))

    #model = models.get_model(width, height, channels, n_classes, hl=6, max_pool=False, nnodes=220)
    model = models.get_vgg_model2(width, height, channels, n_classes, hl=6, max_pool=False, nnodes=220)
    model = models.compile_model(model)
    n_epochs = 50
    history = model.fit(x_train, y_train, batch_size=20, epochs=n_epochs)
    test_scores = model.evaluate(x_test, y_test, verbose=0)
    test_scores.append(history.history['sparse_categorical_accuracy'][n_epochs - 1])
    results.append(test_scores)
    tf.keras.backend.clear_session()
    iteration_end_time = time.time()
    iteration_time = (iteration_end_time - iteration_start_time) / 60
    total_elapsed_time = (iteration_end_time - start_time) / 60
    est_complete = (total_elapsed_time / (idx + 1)) * (num_iterations - idx)
    print(f'Iteration Time:      {iteration_time}')
    print(f'Total Elapsed Time:  {total_elapsed_time}')
    print(f'Est. Time Remaining: {est_complete}')

results = np.array(results)
print(np.average(results, axis=0))
print(f'Total Elapsed Time:  {total_elapsed_time}')

with open('results.txt', 'a+') as f:
    f.write(f'''{num_subjects} images, n_epochs {n_epochs}, Validation Set Size: {validation_size}:
        {np.average(results, axis=0)} \n''')
    f.write('----------------------------------------------------\n\n')






