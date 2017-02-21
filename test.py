import preprocess
import generator
import matplotlib.pyplot as plt

import csv
import os

import numpy as np
import pandas
from sklearn.model_selection import train_test_split

from generator import generator_validation, generator_train
import model
import preprocess

def read_samples():
    samples = []
    with open('./data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    # remove header
    samples.pop(0)
    return samples

samples = read_samples()
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# img, steer = preprocess.preprocess_image_file_train(['IMG/center_2016_12_01_13_30_48_287.jpg', ' IMG/left_2016_12_01_13_30_48_287.jpg', ' IMG/right_2016_12_01_13_30_48_287.jpg', ' 0', ' 0',
#  ' 0', ' 22.14829'])

train_generator = generator.generator_validation(train_samples, batch_size=32)

X_train, y_train = next(train_generator)
print(X_train.shape)
print(y_train.shape)

plt.imshow(X_train[0])
print(y_train[0])
plt.show()