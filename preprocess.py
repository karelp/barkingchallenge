import glob
import os
import random

import librosa
import numpy as np

sample_length = 2
sample_spacing = 0.3
noise_sample_spacing = sample_spacing * 4

# Set the path to the full dataset
fulldatasetpath = 'i:/BarkingDataset/'
positive_files = 'bark'
negative_files = 'noise'


def silence_ratio(segment, silence_threshold):
    return np.sum(np.abs(segment) < silence_threshold) / len(segment)


def extract_features(file_name, sample_spacing, ignore_silence):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        spacing = int(sample_spacing * sample_rate)
        max_volume = np.max(audio)
        while len(audio) < sample_length * sample_rate:  # repeat sounds that are too short
            audio = np.repeat(audio, 2)
        result = []
        for i in range(0, len(audio), spacing):
            if i + int(sample_length) * sample_rate < len(audio):
                segment = audio[i:i + int(sample_length) * sample_rate]
                if (not ignore_silence) or silence_ratio(segment, 0.03 * max_volume) <= 0.9:  # do not add gaps
                    result.append(segment)
        return result

    except Exception as e:
        print("Error encountered while parsing file: ", file_name, e)
        return None


features = []

positive_files = glob.glob(os.path.join(fulldatasetpath, positive_files, "*.wav"))
negative_files = glob.glob(os.path.join(fulldatasetpath, negative_files, "*.wav"))

# Iterate through each sound file and extract the features
X = []
y = []
for class_label, files in enumerate([negative_files, positive_files]):
    for file_name in files:
        # for barks sample all subintervals, for noise sample fewer to make the dataset more balanced
        data = extract_features(file_name, sample_spacing=sample_spacing if class_label == 1 else noise_sample_spacing, ignore_silence=class_label == 1)

        for datapoint in data:
            X.append(datapoint)
            y.append(class_label)


print('Finished feature extraction from ', len(negative_files) + len(positive_files), ' files')

# Convert features and corresponding classification labels into numpy arrays
X = np.array(X)
y = np.array(y)

print("Positive features percentage: ", np.sum(y) / len(y))

# split the dataset
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1337)
num_labels = y.shape[0]

np.save("x_train.npy", x_train)
np.save("x_test.npy", x_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

# np.save("x_train_mfcc.npy", x_train)
# np.save("x_test_mfcc.npy", x_test)
# np.save("y_train_mfcc.npy", y_train)
# np.save("y_test_mfcc.npy", y_test)
np.save("labels.npy", ["noise", "bark"])

