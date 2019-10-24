'''
For Loading images CK+ and JAFFE
Returns a list of Image objects

@author austin
'''

import os
import re
import sys
from logger import Logger, about as a1
import sys
import numpy as np
from images import CkImage, about as a2, JaffeImage


version = '1.0.0'

change_history = '''
date       author    version    message
###############################################################
09/25/19   austin    1.0.0      Initial version
###############################################################
'''

DATA_DIR = os.environ['DATA_DIR']
#PROJECT_DIR = os.environ['PROJ_DIR']


def about():
    print('-------------------------------------------------')
    print('Name:', os.path.basename(__file__))
    print('            Change History',change_history)
    print('Current Version:', version)
    print('-------------------------------------------------')
    a1()
    a2()



def load_jaffe_images(height, width, channels, face=True, edge=True, eyes=False):
    if sys.platform == 'win32':
        IMG_DIR = DATA_DIR + '/jaffedbase/jaffe'
    else:
        IMG_DIR = DATA_DIR + '/jaffe'
    logger = Logger(debug=1)
    tiff_pattern = re.compile('\.tiff', re.IGNORECASE)

    images = []
    for file_name in os.listdir(IMG_DIR):
        if tiff_pattern.search(file_name):
            img = JaffeImage(logger)
            ret = img.get_jaffe_image(IMG_DIR, file_name)
            #logger.put_msg('D', str(ret), name='Main')
            if ret:
                try:
                    img.load_image_from_file(height=height, width=width, face=face, edge=edge, eyes=eyes)
                except Exception as e:
                    print(e)
                    continue
                img.class_name = img.class_labels[img.emotion_class]
                img.img_array = img.img_array[..., np.newaxis]
                images.append(img)
    return images




def load_ck_images(height, width, channels):
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
                img.load_image_from_file(height=height, width=width)
                img.class_name = LABEL_NAMES[img.emotion_class]
                img.img_array = img.img_array[..., np.newaxis]
                images.append(img)
    return images



