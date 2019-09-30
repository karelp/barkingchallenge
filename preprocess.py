import numpy as np
import pandas as pd
import os
import librosa

sample_length = 2
sample_spacing = 0.5


def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        spacing = int(sample_spacing * sample_rate)
        result = []
        for i in range(0, len(audio), spacing):
            if i + int(sample_length) * sample_rate < len(audio):
                mfccs = librosa.feature.mfcc(y=audio[i:i + int(sample_length) * sample_rate], sr=sample_rate, n_mfcc=60)
                # mfccsscaled = np.mean(mfccs.T, axis=0)
                result.append(mfccs)
        return result


    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None


# Load various imports


# Set the path to the full UrbanSound dataset
fulldatasetpath = 'j:/UrbanSound8K/audio/'

metadata = pd.read_csv(fulldatasetpath + '../metadata/UrbanSound8K.csv')

features = []

# Iterate through each sound file and extract the features
for index, row in metadata.iterrows():
    file_name = os.path.join(os.path.abspath(fulldatasetpath), 'fold' + str(row["fold"]) + '/',
                             str(row["slice_file_name"]))

    class_label = row["class"]
    data = extract_features(file_name)

    for datapoint in data:
        features.append([datapoint, class_label])

# Convert into a Panda dataframe
featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])

print('Finished feature extraction from ', len(featuresdf), ' files')

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

# Convert features and corresponding classification labels into numpy arrays
X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())

# Encode the classification labels
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))

# split the dataset
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state=1337)
num_labels = yy.shape[1]

np.save("x_train_mfcc.npy", x_train)
np.save("x_test_mfcc.npy", x_test)
np.save("y_train_mfcc.npy", y_train)
np.save("y_test_mfcc.npy", y_test)
np.save("num_labels.npy", np.array([num_labels], dtype=np.int))
np.save("labels.npy", le.classes_)

