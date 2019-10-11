import os

from kapre.augmentation import AdditiveNoise
from kapre.time_frequency import Melspectrogram, Spectrogram
from kapre.utils import Normalization2D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from datetime import datetime
import keras
import numpy as np
from keras import Model, Input
from keras.engine import InputLayer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

x_train = np.load("x_train.npy")[:, np.newaxis, ...]
x_test = np.load("x_test.npy")[:, np.newaxis, ...]
y_train = np.load("y_train.npy")[:, np.newaxis, np.newaxis, np.newaxis, ...]
y_test = np.load("y_test.npy")[:, np.newaxis, np.newaxis, np.newaxis, ...]

# x_train = np.load("x_train_mfcc.npy")[..., np.newaxis]
# x_test = np.load("x_test_mfcc.npy")[..., np.newaxis]
# y_train = np.load("y_train_mfcc.npy")[:, np.newaxis, np.newaxis, np.newaxis, ...]
# y_test = np.load("y_test_mfcc.npy")[:, np.newaxis, np.newaxis, np.newaxis, ...]


from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, GlobalAveragePooling1D, LeakyReLU, AveragePooling1D, \
    MaxPooling2D, Conv2D, AveragePooling2D, BatchNormalization

labels = np.load("labels.npy")
num_labels = 1  # len(labels)
filter_size = 3
activation = lambda: LeakyReLU(alpha=0.001)


# Construct model
def model_raw_sound(x_train, num_labels):
    model_input = x = Input(shape=x_train[0].shape)
    x = Conv1D(filters=16, kernel_size=filter_size)(x)
    x = activation()(x)
    x = MaxPooling1D(pool_size=4)(x)
    x = Dropout(0.2)(x)

    x = Conv1D(filters=32, kernel_size=filter_size)(x)
    x = activation()(x)
    x = MaxPooling1D(pool_size=4)(x)
    x = Dropout(0.2)(x)

    x = Conv1D(filters=64, kernel_size=filter_size)(x)
    x = activation()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    x = Conv1D(filters=128, kernel_size=filter_size)(x)
    x = activation()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    x = AveragePooling1D(pool_size=(int(x.get_shape()[1]),))(x)

    x = Conv1D(filters=num_labels, kernel_size=1, padding='valid', activation='softmax' if num_labels > 1 else 'sigmoid')(x)

    model = Model(inputs=[model_input], outputs=[x])

    model.summary()
    return model


def model_mfcc(x_train, num_labels):
    model_input = x = Input(shape=x_train[0].shape)
    x = Conv2D(filters=16, kernel_size=filter_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = activation()(x)
    x = MaxPooling2D(pool_size=2)(x)

    x = Conv2D(filters=32, kernel_size=filter_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = activation()(x)
    x = MaxPooling2D(pool_size=2)(x)

    x = Conv2D(filters=64, kernel_size=filter_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = activation()(x)
    x = MaxPooling2D(pool_size=2)(x)

    x = Conv2D(filters=128, kernel_size=filter_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = activation()(x)
    x = MaxPooling2D(pool_size=2)(x)

    x = Conv2D(filters=256, kernel_size=filter_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = activation()(x)

    x = AveragePooling2D(pool_size=(int(x.get_shape()[1]), int(x.get_shape()[2])))(x)

    x = Conv2D(filters=num_labels, kernel_size=1, padding='valid', activation='softmax' if num_labels > 1 else 'sigmoid')(x)

    model = Model(inputs=[model_input], outputs=[x])

    model.summary()
    return model


def model_mfcc_layer(x_train, num_labels):
    sr = 22050
    model_input = x = Input(shape=x_train[0].shape)
    x = Melspectrogram(n_dft=512, n_hop=sr // 128 + 1,
                   padding='same', sr=sr, n_mels=128,
                   fmin=0.0, fmax=sr / 2, power_melgram=2.0,
                   return_decibel_melgram=True, trainable_fb=False,
                   trainable_kernel=False,
                   name='trainable_stft')(x)
    # x = Spectrogram(n_dft=512, n_hop=sr // 128 + 1,
    #       return_decibel_spectrogram=False, power_spectrogram=2.0,
    #       trainable_kernel=False, name='static_stft')(x)
    # x = Normalization2D(str_axis='freq')(x)
    # x = AdditiveNoise(power=0.3)(x)
    x = Conv2D(filters=16, kernel_size=filter_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = activation()(x)
    x = MaxPooling2D(pool_size=2)(x)

    x = Conv2D(filters=32, kernel_size=filter_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = activation()(x)
    x = MaxPooling2D(pool_size=2)(x)

    x = Conv2D(filters=64, kernel_size=filter_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = activation()(x)
    x = MaxPooling2D(pool_size=2)(x)

    x = Conv2D(filters=128, kernel_size=filter_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = activation()(x)
    x = MaxPooling2D(pool_size=2)(x)

    x = Conv2D(filters=256, kernel_size=filter_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = activation()(x)

    x = AveragePooling2D(pool_size=(int(x.get_shape()[1]), int(x.get_shape()[2])))(x)

    x = Conv2D(filters=num_labels, kernel_size=1, padding='valid', activation='softmax' if num_labels > 1 else 'relu')(x)

    model = Model(inputs=[model_input], outputs=[x])

    model.summary()
    return model


# model = model_raw_sound(x_train, num_labels)
model = model_mfcc_layer(x_train, num_labels)
model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss=keras.losses.categorical_crossentropy if num_labels > 1 else keras.losses.mean_squared_error, metrics=['accuracy'])

num_epochs = 256
num_batch_size = 64

checkpointer = ModelCheckpoint(filepath='saved_models/weights.hdf5',
                               verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=4, min_lr=0.0000001, verbose=1)
start = datetime.now()

try:
    model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[checkpointer, reduce_lr], verbose=1)
except KeyboardInterrupt:
    print("Training interrupted by the user")

model.save("final_model.h5")


duration = datetime.now() - start
print("Training completed in time: ", duration)
