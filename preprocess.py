import os
import pandas as pd
import numpy as np
from scipy import signal
import cv2
import math

rows = 160
cols = 320

def add_random_shadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image

def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = .25 + np.random.uniform()
    #print(random_bright)
    image1[:, :, 2] = image1[:, :, 2]*random_bright
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1

def trans_image(image, steer,trans_range):
    # Translation
    tr_x = trans_range * np.random.uniform() - trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 40 * np.random.uniform() - 40/2
    #tr_y = 0
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    image_tr = cv2.warpAffine(image, Trans_M, (cols,rows))
    
    return image_tr, steer_ang, tr_x

def cropImage(image, new_shape=(200, 66)):
    # Preprocessing image files
    shape = image.shape
    # note: numpy arrays are (row, col)!
    image = image[math.floor(shape[0]/5):shape[0]-25, 0:shape[1]]
    image = cv2.resize(image, new_shape, interpolation=cv2.INTER_AREA)
    return image

def preprocess_image_file_train(line_data):
    # Preprocessing training files and augmenting
    i_lrc = np.random.randint(3)
    if i_lrc == 0:
        path_file = line_data[0].strip()
        shift_ang = .25
    if i_lrc == 1:
        path_file = line_data[1].strip()
        shift_ang = 0.
    if i_lrc == 2:
        path_file = line_data[2].strip()
        shift_ang = -.25
    y_steer = float(line_data[3].strip()) + shift_ang
    image = cv2.imread('/input/' + path_file.replace(' ', ''))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = add_random_shadow(image)
    image,y_steer,tr_x = trans_image(image, y_steer, 150)
    image = augment_brightness_camera_images(image)
    image = cropImage(image)
    image = np.array(image)
    ind_flip = np.random.randint(2)
    if ind_flip == 0:
        image = cv2.flip(image, 1)
        y_steer = -y_steer
    return image, y_steer

def preprocess_image_file_predict(line_data):
    # Preprocessing Prediction files and augmenting
    path_file = line_data[0].strip()
    y_steer = float(line_data[3].strip())
    image = cv2.imread('/input/' + path_file.replace(' ', ''))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cropImage(image)
    image = np.array(image)
    return image, y_steer
