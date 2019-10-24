'''
Image classes

@author austin
'''

import os
import cv2
import re
from logger import Logger
import sys
import argparse
import numpy as np

version = '1.0.0'

change_history = '''
date       author    version    message
###############################################################
09/25/19   austin    1.0.0      Initial version
10/15/19   austin    1.1.0      Adding face detection
###############################################################
'''

DATA_DIR = os.environ['DATA_DIR']
PROJECT_DIR = os.environ['PROJ_DIR']

CASCADE_DIR = PROJECT_DIR + '/FaceDetection/xml'
FACE_XML = CASCADE_DIR + '/haarcascade_frontalface_default.xml'
EYE_XML = CASCADE_DIR + '/haarcascade_eye.xml'


FACE_CASCADE = cv2.CascadeClassifier(FACE_XML)
EYE_CASCADE = cv2.CascadeClassifier(EYE_XML)

def about():
    print('-------------------------------------------------')
    print('Name:', os.path.basename(__file__))
    print('            Change History',change_history)
    print('Current Version:', version)
    print('-------------------------------------------------')

def select_eyes(eyes):
    if len(eyes) == 0:
        return None
    if len(eyes) > 2:
        #eyes = top_eyes(eyes)
        indxs = []
        for eye in eyes:
            indxs.append(eye[1])

        val, idx1 = min((val, idx) for (idx, val) in enumerate(indxs))
        indxs[idx1] = max(indxs)
        val, idx2 = min((val, idx) for (idx, val) in enumerate(indxs))

        tmp = [eyes[idx1], eyes[idx2]]
        if tmp[0][0] > tmp[1][0]:
            tmp2 = tmp
            tmp = [tmp2[1], tmp2[0]]
        eyes = tmp

    if len(eyes) == 2:
        if eyes[0][0] > eyes[1][0]:
                tmp2 = eyes
                eyes = [tmp2[1], tmp2[0]]
        return eyes

    if len(eyes) < 2:
        return None

class Image():
    '''Store image information'''
    def __init__(self, logger=None):
        self.img_array = None
        self.emotion_class = None
        self.image_file_path = None
        self.image_file_name = None
        self.shape = None
        self.image_dir = None
        self.class_name = None
        if logger is None:
            self.logger = Logger()
        else:
            self.logger = logger

    def load_image_from_file(self, height=None, width=None, chan=None, read_type=cv2.IMREAD_GRAYSCALE, face=False, edge=False, eyes=False):
        '''Load the image using opencv imread. Deaults to IMREAD_GRAYSCALE
        Populates the image array from the image file.
        Raises an exception if the file is not defined.
        '''
        full_path_to_image = f'{self.image_file_path}/{self.image_file_name}'
        if not os.path.isfile(full_path_to_image):
            raise Exception('Not a file', full_path_to_image)
        img = cv2.imread(full_path_to_image, read_type)


        if face:
            face = FACE_CASCADE.detectMultiScale(img, 1.3, 5)
            for (x,y,w,h) in face:
                img = img[y:y+h, x:x+w]

        if None not in [height, width]:
            img = cv2.resize(img, (height, width))
        else:
            height = 128
            width = 128

        if eyes:
            eyes = EYE_CASCADE.detectMultiScale(img)
            eyes = select_eyes(eyes)
            if eyes is None:
                raise Exception('Eyes not found')
            tmp_eyes = []
            for (ex,ey,ew,eh) in eyes:
                eye_img = img[ey:ey+eh, ex:ex+ew]
                eye_img = cv2.resize(eye_img, (int(height/2), int(width/2)))
                tmp_eyes.append(eye_img)
            eye_img = np.concatenate((tmp_eyes[0], tmp_eyes[1]), axis=0)
            img = np.concatenate((eye_img, img), axis=1)

        if edge:
            #img = cv2.Laplacian(img, cv2.CV_64F)
            img = cv2.Canny(img, 80, 200)

        img = cv2.resize(img, (height, width))

        self.img_array = img

        self.shape = self.img_array.shape


    def save_image(self, path=''):
        cv2.imwrite(path + '' + self.image_file_name, self.img_array)


    def show_image(self, name=None):
        '''Use opencv to show the image.
        Defaults to the file name.
        Returns None if there is no image array'''
        if self.img_array is None:
            return None

        if name is None:
            name = self.image_file_name

        self.logger.put_msg('D', f'File Name: {self.image_file_name} Class: {self.emotion_class} Label: {self.class_name}', name='Image')
        cv2.imshow(name, self.img_array)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

class JaffeImage(Image):
    '''Class for the JAFFE images'''

    def __init__(self, logger=None):
        super().__init__(logger)

    def get_jaffe_image(self, image_file_path, image_file_name):
        '''Load the images from their diretory'''
        self.image_file_path = image_file_path
        self.image_file_name = image_file_name
        patterns, image_labels = self.get_patterns()
        self.class_labels = image_labels
        file_name_ptr = re.compile('([A-Za-z][A-Za-z])\.([A-Za-z][A-Za-z]\d)\.(\d+)\.tiff')
        full_file_path = f'{self.image_file_path}/{self.image_file_name}'
        if os.path.isfile(full_file_path):
            name_m = re.search(file_name_ptr, self.image_file_name)
            if name_m:
                self.subject_id = name_m.group(1)
            else:
                return 0

            for idx, pattern in enumerate(patterns):
                m = pattern.search(self.image_file_name)
                if m:
                    self.emotion_class = idx
                    return 1
            return 0
        else:
            return 0


    def get_patterns(self, pattern_list=None):
        '''Build the list of patterns.
        Requires a list of pattern names'''
        pattern_ref = {
            'HA': re.compile('HA'),
            'SA': re.compile('SA'),
            'SU': re.compile('SU'),
            'AN': re.compile('AN'),
            'DI': re.compile('DI'),
            'FE': re.compile('FE'),
            'NE': re.compile('NE')
        }
        patterns = []
        labels = []
        if pattern_list is None:
            for patttern_name, pattern in pattern_ref.items():
                patterns.append(pattern)
                labels.append(patttern_name)
        else:
            for patttern_name in pattern_list:
                try:
                    patterns.append(pattern_ref[patttern_name])
                except KeyError as e:
                    self.logger.put_msg('E', f'Key error {patttern_name} in get_patterns.', name='images.py')
                    raise e
                labels.append(patttern_name)
        return patterns, labels



class CkImage(Image):
    '''Class for CK+ images'''
    def __init__(self, logger=None):
        super().__init__(logger)
        self.emotion_dir = None
        self.emotion_file_name = None

    def get_ck_img(self, emotion_dir, emotion_file, image_dir, subject_id, inst_dir):
        '''Load the image information based on the emotion file.
        Used because images are loaded by scanning the emotion files'''
        self.image_dir = image_dir
        self.inst_dir = inst_dir
        self.subject_id = subject_id

        self.emotion_dir = emotion_dir
        self.emotion_file_name = emotion_file

        self.image_file_name = re.sub('_emotion.txt', '.png', self.emotion_file_name)

        self.image_file_path = f'{self.image_dir}/{self.subject_id}/{self.inst_dir}'
        self.emotion_file_path = f'{self.emotion_dir}/{self.subject_id}/{self.inst_dir}'

        full_file_path = f'{self.emotion_file_path}/{self.emotion_file_name}'
        if os.path.isfile(full_file_path):
            with open(full_file_path, 'r') as f:
                x = f.readline()
                self.emotion_class = int(eval(x))
            return 1
        else:
            return 0