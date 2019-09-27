'''
Driver

@author austin
'''

import data
import os
from logger import Logger
import numpy as np
import models

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
images = data.load_ck_images(height, width, channels)

logger.put_msg("I", f"Processing {len(images)} images")


x = lambda a: [a.subject_id, a.emotion_class, a.img_array]

def lcd(num):
    for i in range(3, 9):
        if num % i == 0:
            return i

subjects = map(x, images)
data = np.array(list(subjects))
num_subjects = len(data)
validation_size = lcd(num_subjects)
logger.put_msg("I", f"Validation Set Size: {validation_size}")
num_iterations = int(num_subjects / validation_size)

start = 0
end = 0
results = []
for idx in range(num_iterations):
    start = idx * validation_size
    end = start + validation_size
    test = data[start:end, :]
    print(test.shape)

    x_test = test[:,2]
    print(x_test[0].shape)
    x_test = np.reshape(x_test, (validation_size, height, width, channels))
    print(x_test.shape)
    exit()
    y_test = test[:,1]
    train = np.concatenate((data[0:start,:], data[end:,:]), axis=0)
    x_train = train[:,2]
    y_train = train[:,1]
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    print(x_train[0])
    exit()
    n_classes = len(np.unique(y_train))
    model = models.get_model(width, height, channels, n_classes)
    model = models.compile_model(model)
    model.fit(x_train, y_train, batch_size=1, epochs=10)
    test_scores = model.evaluate(x_test, y_test, verbose=0)
    results.append(test_scores)

results = np.array(results)
print(np.average(results, axis=0))







