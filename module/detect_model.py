# noinspection PyUnresolvedReferences
from tensorflow.keras.models import Model
# noinspection PyUnresolvedReferences
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout


def detect_model():
    inputs = Input((256, 256, 3))
    conv1 = Conv2D(96, (11, 11), strides=(4, 4), activation='relu', padding='valid')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv1)
    conv2 = Conv2D(256, (11, 11), strides=(1, 1), activation='relu', padding='valid')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv2)
    conv3 = Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding='valid')(pool2)
    conv4 = Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding='valid')(conv3)
    conv5 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid')(conv4)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv5)
    flat = Flatten()(pool3)
    dense1 = Dense(4096, activation='relu')(flat)
    dropout1 = Dropout(0.4)(dense1)
    output1 = Dense(4, activation='sigmoid')(dropout1)
    model = Model(inputs=[inputs], outputs=[output1])
    return model
