import csv
import os

import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

import model
import preprocess
from generator import generator_train, generator_validation
from keras.callbacks import ModelCheckpoint
from para import dot


def read_samples():
    samples = []
    with open(dot + '/input/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    # remove header
    samples.pop(0)
    return samples

samples = read_samples()
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
resample_rate = 3
## generate more training data by resampling
train_samples = resample(samples, n_samples=int(len(train_samples)*resample_rate), random_state=42)
print("Number of training samples: {}".format(len(train_samples)))
print("Number of validation samples: {}".format(len(validation_samples)))


# compile and train the model using the generator function
train_generator = generator_train(train_samples, batch_size=128)
validation_generator = generator_validation(validation_samples, batch_size=128)

nvidia_model = model.build_model()
nvidia_model.compile(loss='mse', optimizer='adam')
checkpointer = ModelCheckpoint(filepath=dot + "/output/weights.hdf5", verbose=1, save_best_only=True)
history_object = nvidia_model.fit_generator(train_generator,
                                            samples_per_epoch=len(train_samples),
                                            validation_data=validation_generator,
                                            nb_val_samples=len(validation_samples),
                                            nb_epoch=1000,
                                            callbacks=[checkpointer])

# save the best performance model
nvidia_model.load_weights(dot + '/output/weights.hdf5')
nvidia_model.save(dot + '/output/nvidia_model.h5')



## TODO:
## 1. Random deny low steering value, filter out going straight bias
## 2. preprocess (crop image) for driving (in model?) check
## 3. vertical shift to simulate up/down hill road check
## 4. create artificial shadow to simulate lighting change check
## 5. 
