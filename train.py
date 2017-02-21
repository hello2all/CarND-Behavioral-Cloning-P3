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
print(len(train_samples), len(validation_samples))


# compile and train the model using the generator function
train_generator = generator_train(train_samples, batch_size=32)
validation_generator = generator_validation(validation_samples, batch_size=32)

Nvidia_model = model.build_model()
Nvidia_model.compile(loss='mse', optimizer='adam')
history_object = Nvidia_model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=2)
