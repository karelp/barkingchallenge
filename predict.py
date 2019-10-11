import os

import scipy
from kapre.augmentation import AdditiveNoise
from kapre.time_frequency import Melspectrogram
from kapre.utils import Normalization2D

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import sys

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr
import librosa
import numpy as np
from keras import Model
import json

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def reshape_model(model, new_shape):
    copy = keras.models.model_from_json(model.to_json())
    copy._layers[0].batch_input_shape = (None,) + new_shape
    new_model = keras.models.model_from_json(copy.to_json())
    new_model.set_weights(model.get_weights())
    return new_model


labels = np.load("labels.npy")
model = keras.models.load_model("final_model.h5", custom_objects={'Melspectrogram': Melspectrogram, 'Normalization2D': Normalization2D, 'AdditiveNoise': AdditiveNoise})
dog_bark_index = 0
# model = reshape_model(model, (None, None, 1))

audio, sample_rate = librosa.load(sys.argv[1] if len(sys.argv) > 1 else "Barkio-Barking-Detection-Challange-1.mp3", res_type='kaiser_fast')

# mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=60 * len(audio) // (2 * sample_rate))
# predictions = model.predict(mfccs[np.newaxis, ..., np.newaxis])[0]


step = 0.5
results = []
for i in range(0, len(audio), int(sample_rate * step)):
    input_sample = np.zeros(2 * sample_rate)
    input_sample[0:len(audio[i:i + 2 * sample_rate])] = audio[i:i + 2 * sample_rate]
    # input_sample = librosa.feature.mfcc(y=input_sample, sr=sample_rate, n_mfcc=60)
    results.append(model.predict(input_sample[np.newaxis, np.newaxis, ...])[0])

bark_probabilities = np.zeros(len(audio) // int(sample_rate * step) + 1)
bark_counts = np.zeros_like(bark_probabilities)
for i, r in enumerate(results):
    p = np.squeeze(r)
    if len(p.shape) == 0:
        p = p[np.newaxis, ...]
    bark_probabilities[i] += min(1, p[dog_bark_index])
    # bark_probabilities[i] = max(bark_probabilities[i], min(1, p[dog_bark_index]))
    bark_counts[i] += 1
    if i + 1 < len(bark_probabilities):
        bark_probabilities[i + 1] += min(1, p[dog_bark_index])
        # bark_probabilities[i + 1] = max(bark_probabilities[i + 1], min(1, p[dog_bark_index]))
        bark_counts[i + 1] += 1
bark_probabilities /= bark_counts

bark_probabilities = scipy.signal.medfilt(bark_probabilities, 3)

current_start = 0
threshold = 0.8
barking = False
ranges = []
for i, p in enumerate(bark_probabilities):
    if p > threshold and not barking:
        current_start = i
        barking = True
    elif p <= threshold and barking:
        ranges.append({"start": int(current_start * step * 1000), "duration": int((i - current_start) * step * 1000)})
        barking = False
if barking:
    ranges.append({"start": int(current_start * step * 1000), "duration": int((len(bark_probabilities) - current_start) * step * 1000)})

print(json.dumps(ranges))
