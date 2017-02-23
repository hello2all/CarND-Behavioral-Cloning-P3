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
train_generator = generator_train(train_samples, batch_size=128)
validation_generator = generator_validation(validation_samples, batch_size=128)

nvidia_model = model.build_model()
nvidia_model.compile(loss='mse', optimizer='adam')
history_object = nvidia_model.fit_generator(train_generator,
                                            samples_per_epoch=len(train_samples),
                                            validation_data=validation_generator,
                                            nb_val_samples=len(validation_samples),
                                            nb_epoch=100)

nvidia_model.save('./output/nvidia_model.h5')


## TODO:
## 1. Random deny low steering value, filter out going straight bias
## 2. preprocess (crop image) for driving (in model?) check
## 3. vertical shift to simulate up/down hill road
## 4. create artificial shadow to simulate lighting change
## 5. 