from keras.callbacks import Callback
from keras.models import model_from_json, Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D

def build_model(cameraFormat=(66, 200, 3)):
    """
    Build and return a CNN; details in the comments.
    The intent is a scaled down version of the model from "End to End Learning
    for Self-Driving Cars": https://arxiv.org/abs/1604.07316.
    Args:
    cameraFormat: (3-tuple) Ints to specify the input dimensions (color
    channels, rows, columns).
    Returns:
    A compiled Keras model.
    """
    print("Building model...")
    row, col, ch = cameraFormat # Trimmed image format

    model = Sequential()

    # Use a lambda layer to normalize the input data
    model.add(Lambda(
        lambda x: x/127.5 - 1.,
        input_shape=(row, col, ch),
        output_shape=(row, col, ch))
    )

    # Several convolutional layers, each followed by ELU activation
    # 8x8 convolution (kernel) with 4x4 stride over 16 output filters
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    # 5x5 convolution (kernel) with 2x2 stride over 32 output filters
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    # 5x5 convolution (kernel) with 2x2 stride over 64 output filters
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    # Flatten the input to the next layer
    model.add(Flatten())
    # Apply dropout to reduce overfitting
    model.add(Dropout(.2))
    model.add(ELU())
    # Fully connected layer
    model.add(Dense(512))
    # More dropout
    model.add(Dropout(.5))
    model.add(ELU())
    # Fully connected layer with one output dimension (representing the speed).
    model.add(Dense(1))

    # Adam optimizer is a standard, efficient SGD optimization method
    # Loss function is mean squared error, standard for regression problems
    model.compile(optimizer="adam", loss="mse")

    return model