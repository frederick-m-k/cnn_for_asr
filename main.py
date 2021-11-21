
# imports
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import librosa
import json
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout
from tensorflow.keras.utils import to_categorical

from sklearn.utils.class_weight import compute_class_weight

###### global variables
corpus_path = "resources/phattsessionz/"
rec2mfccs = dict()
phoneme2mfccs = dict()

# methods


def get_librosa_mfccs(wav_path, n_mfccs=13, hop_length=128):
    '''
        n_mfccs: defines the number of mfccs
        hop_length: defines the amount of samples between two windows

        return a list of mfccs:
        [
            [mfcc1_frame1, mfcc1_frame2, ...],
            [mfcc2_frame2, mfcc2_frame2, ...],
            ...
        ]
    '''
    y, sr = librosa.load(wav_path)  # , dtype=float)
    x = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfccs, hop_length=hop_length)
    return x


def get_librosa_duration(wav):
    return librosa.get_duration(filename=wav)


def npArrayToList(ndarray):
    ''' array should come in the following format:
            [
                [all mfccs 1],
                [all mfccs 2], 
                ...
            ]
        this method returns the same data in the following format:
            [
                [frame 1 with all mfccs],
                [frame 2 with all mfccs],
                ...
            ]'''
    listStructure = []
    for array in ndarray:
        listStructure.append(np.ndarray.tolist(array))

    result = []
    allMFCCS = []
    for frame in range(0, len(listStructure[0])):
        for mfcc in range(0, len(listStructure)):
            allMFCCS.append(listStructure[mfcc][frame])
        result.append(allMFCCS)
        allMFCCS = []
    return result


def parseTextGrid_withPhonem(textGrid):
    ''' returns an textgrid and its information as a list of tripel. Those hold the phonem label, the start and the end time'''
    times = []
    counter = 0
    secondCounter = 0
    foundWortTier = False
    tempStart = 0
    tempEnd = 0
    if os.path.isfile(textGrid):
        with open(textGrid, "r") as openFile:
            try:
                lines = openFile.readlines()
            except UnicodeDecodeError:
                with open(textGrid, "r", encoding="utf-16-be") as oF:
                    lines = oF.readlines()
            for line in lines:
                if "name = \"MAU\"" in line:
                    foundWortTier = True
                elif "name = " in line:
                    foundWortTier = False
                if foundWortTier:
                    if "xmin =" in line:
                        tempStart = line.split("=")[1].strip()
                        secondCounter += 1
                    elif "xmax =" in line:
                        tempEnd = line.split("=")[1].strip()
                        secondCounter += 1
                    elif "text = " in line:
                        if "text = \"\"" in line or "<p:>" in line:
                            secondCounter = 0
                        else:
                            secondCounter += 1
                            phoneme = line.split("=")[1].strip("\" \n")
                    else:
                        secondCounter = 0
                    if (secondCounter == 3):
                        times.append(
                            (phoneme, float(tempStart), float(tempEnd)))
                        counter += 1
                        secondCounter = 0
    return times


def gather_all_mfccs():
    for sub in sorted(os.listdir(corpus_path)):
        sub_folder = corpus_path + sub
        if os.path.isdir(sub_folder):
            for file_name in os.listdir(sub_folder):
                one_file = sub_folder + '/' + file_name
                if os.path.isfile(one_file) and not file_name.startswith('.'):
                    if file_name.endswith(".wav"):  # found wav file
                        mfccs = npArrayToList(get_librosa_mfccs(one_file))
                        rec2mfccs[file_name.replace(".wav", '')] = mfccs


def gather_all_tgs():
    for sub in sorted(os.listdir(corpus_path)):
        sub_folder = corpus_path + sub
        if os.path.isdir(sub_folder):
            for file_name in os.listdir(sub_folder):
                one_file = sub_folder + '/' + file_name
                if os.path.isfile(one_file) and not file_name.startswith('.'):
                    if file_name.endswith(".TextGrid") and file_name.replace(".TextGrid", '') in rec2mfccs.keys():
                        tg_data = parseTextGrid_withPhonem(one_file)
                        merge_frames_tg_sepPhones(rec2mfccs[file_name.replace(
                            ".TextGrid", '')], tg_data, one_file)


def merge_frames_tg_allInOne(mfcc_frames, tg_data, whole_file_path):
    wav_length = get_librosa_duration(
        whole_file_path.replace(".TextGrid", ".wav"))
    distance = wav_length / len(mfcc_frames)
    counter = 0
    for frame in mfcc_frames:
        time_point = counter * distance

        for (label, start, end) in tg_data:
            if time_point >= start and time_point <= end:
                # found correct segment
                if not label in phoneme2mfccs.keys():
                    phoneme2mfccs[label] = []
                phoneme2mfccs[label].append(frame)
                break
        counter += 1


def merge_frames_tg_sepPhones(mfcc_frames, tg_data, whole_file_path):
    wav_length = get_librosa_duration(
        whole_file_path.replace(".TextGrid", ".wav"))
    distance = wav_length / len(mfcc_frames)
    counter = 0
    old_label = ""
    frames = []
    for frame in mfcc_frames:
        time_point = counter * distance

        for (label, start, end) in tg_data:
            if time_point >= start and time_point <= end:
                # found correct segment
                if not label in phoneme2mfccs.keys():
                    phoneme2mfccs[label] = []

                if old_label != label:
                    if old_label != "":
                        phoneme2mfccs[old_label].append(frames)
                    old_label = label
                    frames = []
                frames.append(frame)
                break
        counter += 1
    phoneme2mfccs[label].append(frames)


def save_mfccs():
    with open(corpus_path + "librosa_phoneme2mfccs.json", 'w', encoding="utf-8") as dumping:
        json.dump(phoneme2mfccs, dumping)


def load_mfccs():
    with open(corpus_path + "librosa_phoneme2mfccs.json", 'r', encoding="utf-8") as loading:
        phoneme2mfccs = json.load(loading)
    return phoneme2mfccs


def norm_frames(one_phoneme, shortest_phoneme):
    '''
        split a phoneme of length x to smaller parts, each of length shortest_phoneme
        this method calls a recursive one
    '''
    normed_phoneme = []
    if len(one_phoneme) < shortest_phoneme:
        return []
    if len(one_phoneme) == shortest_phoneme:
        return [one_phoneme]
    elif len(one_phoneme) > shortest_phoneme:
        normed_phoneme = _norm_frames(
            0, len(one_phoneme), shortest_phoneme, one_phoneme, normed_phoneme)
    return normed_phoneme


def _norm_frames(missing_start, missing_end, shortest_phoneme, phoneme, normed_phoneme):
    '''
        recursive method
    '''
    if missing_start >= missing_end:  # nothing more to add
        return normed_phoneme

    elif missing_end - missing_start <= shortest_phoneme:  # one missing element is also present here
        further_miss_distance = shortest_phoneme - \
            (missing_end - missing_start)
        start_point = missing_start - int(further_miss_distance / 2)
        end_point = missing_end + int(further_miss_distance / 2)
        if further_miss_distance % 2 != 0:
            end_point += 1

        normed_phoneme.append(phoneme[start_point:end_point])
        return normed_phoneme

    else:  # now we know, that there are at least two more parts of the phoneme missing
        first_end = missing_start + shortest_phoneme
        last_start = missing_end - shortest_phoneme

        first_part_of_phoneme = phoneme[missing_start:first_end]
        normed_phoneme.append(first_part_of_phoneme)

        normed_phoneme = _norm_frames(first_end, last_start,
                                      shortest_phoneme, phoneme, normed_phoneme)

        last_part_of_phoneme = phoneme[last_start:missing_end]
        normed_phoneme.append(last_part_of_phoneme)
        return normed_phoneme


def shape_mfccs_for_training(all_mfccs):
    '''
        input:
            mfccs as dict of phonemes paired with their mfccs as shape (frames by number of coefficients)

        output:
            X: samples of mfccs as shape (number of coeffients by frames)
            y: categorial one-hot vectors of the phoneme labels
    '''
    X = []
    y = []

    label_list = list(all_mfccs.keys())

    shortest_phoneme = sys.maxsize
    for label, phonemes in all_mfccs.items():
        for frames in phonemes:
            shortest_phoneme = min(len(frames), shortest_phoneme)

    shortest_phoneme = 15
    for label, phonemes in all_mfccs.items():
        for one_phoneme in phonemes:
            normed_phonemes = norm_frames(one_phoneme, shortest_phoneme)
            for normed_phoneme in normed_phonemes:
                normed_phoneme = np.array(normed_phoneme).T
                X.append(normed_phoneme)
                y.append(label_list.index(label))

    X = np.array(X)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    y = to_categorical(y, num_classes=len(all_mfccs.keys()))
    return X, y


def get_conv_model(input_shape, output_classes):
    '''
        define a convolutional keras model
        compile and return it
    '''
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation="relu", strides=(1, 1),
                     padding="same", input_shape=input_shape))  # padding="same",
    model.add(Conv2D(32, (3, 3), activation="relu", strides=(1, 1),  # ))
                     padding="same"))
    model.add(Conv2D(64, (3, 3), activation="relu", strides=(1, 1),  # ))
                     padding="same"))
    model.add(Conv2D(128, (3, 3), activation="relu", strides=(1, 1),  # ))
                     padding="same"))
    model.add(MaxPool2D((3, 3)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    # Dense layers are the fully-connected layers
    # model.add(Dense(128, activation="relu"))
    # model.add(Dropout(0.25))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(output_classes, activation="sigmoid"))

    model.summary()

    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["acc"])

    return model


# plotting
def plot_bar_chart():
    data = []
    labels = []
    for phoneme, mfccs in phoneme2mfccs.items():
        labels.append(phoneme)
        data.append(len(mfccs))
    plt.bar(labels, data)
    plt.show()


def plot_amount_per_phoneme(pho2mfccs):
    data = []
    labels = []
    for label, phonemes in pho2mfccs.items():
        labels.append(label)
        data.append(len(phonemes))
    plt.bar(labels, data)
    plt.show()


def plot_meanFrameAmount_per_phoneme(pho2mfccs):
    data = []
    labels = []
    for label, phonemes in pho2mfccs.items():
        labels.append(label)
        _mean = 0
        for phoneme in phonemes:
            _mean += len(phoneme)
        _mean = _mean / len(phonemes)
        data.append(_mean)
    plt.bar(labels, data)
    plt.show()


def main(use_saved):
    if not use_saved:
        gather_all_mfccs()
        gather_all_tgs()
        save_mfccs()
    else:
        pho2mfccs = load_mfccs()

        # plot_amount_per_phoneme(pho2mfccs)
        # plot_meanFrameAmount_per_phoneme(pho2mfccs)
        X, y = shape_mfccs_for_training(pho2mfccs)
        input_shape_net = (X.shape[1], X.shape[2], X.shape[3])

        model = get_conv_model(input_shape_net, len(pho2mfccs.keys()))

        # TODO: maybe to a class weighting (method can be found in sklearn.utils.class_weight: compute_class_weight)
        model.fit(X, y, epochs=10, batch_size=32, shuffle=True)
        # to adapt model, both epochs as well as batch_size can be a good starting point

        model.save("models/first.tf", save_format="tf")


if __name__ == "__main__":
    main(True)
    model = keras.models.load_model("models/first.tf", compile=True)
