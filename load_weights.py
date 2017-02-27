import keras

model = keras.models.load_model('./output/nvidia_model.h5')
model.load_weights('./output/weights.hdf5')
model.save('./output/nvidia_model.h5')

print("loading weights complete!")