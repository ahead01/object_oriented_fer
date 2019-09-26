'''
For Loading images CK+ and JAFFE
Returns a list of Image objects

@author austin
'''

import os
import re
from logger import Logger, about as a1
import sys
import argparse
from images import CkImage, about as a2


version = '1.0.0'

change_history = '''
date       author    version    message
###############################################################
09/25/19   austin    1.0.0      Initial version
###############################################################
'''

DATA_DIR = os.environ['DATA_DIR']
PROJECT_DIR = os.environ['PROJ_DIR']


def about():
    print('-------------------------------------------------')
    print('Name:', os.path.basename(__file__))
    print('            Change History',change_history)
    print('Current Version:', version)
    print('-------------------------------------------------')
    a1()
    a2()


def load_ck_images():
    BASE_DIR = DATA_DIR + '/ck/CK+/'
    EMOTION_DIR = BASE_DIR + 'Emotion_labels/Emotion/'
    IMG_DIR = BASE_DIR + 'extended-cohn-kanade-images/cohn-kanade-images/'
    #logger = Logger(debug=1, log_file='out.txt')
    logger = Logger(debug=0)
    LABEL_NAMES = ['neutral', 'anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

    images = []
    for subject_dir in os.listdir(EMOTION_DIR):
        for inst_dir in os.listdir(EMOTION_DIR + subject_dir + '/'):
            emotion_file_names = os.listdir(EMOTION_DIR + subject_dir + '/' + inst_dir + '/')
            if not emotion_file_names:
                continue
            emotion_file_name = emotion_file_names[0]

            img = CkImage(logger)
            ret = img.get_ck_img(EMOTION_DIR, emotion_file_name, IMG_DIR, subject_dir, inst_dir)
            logger.put_msg('D', str(ret), name='Main')
            if ret:
                img.load_image_from_file(height=128, width=128)
                img.class_name = LABEL_NAMES[img.emotion_class]
                images.append(img)
    return images



