import librosa
import random
import os
import numpy as np
import fnmatch
import tqdm
import tensorflow as tf

"""
This file contains function necessary for working with audio data and input and outputting audio from Wavenet..
"""


# Gets all names of files within a directory
def find_files(directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files

def load_train_valid_filenames(directory, num_samples=None, percent_training=0.9):
    randomized_files = find_files(directory)
    random.shuffle(randomized_files)
    if num_samples is not None:
        randomized_files = randomized_files[:num_samples]
    number_of_training_samples = int(round(percent_training * len(randomized_files)))
    training_files, validation_files = randomized_files[:number_of_training_samples], randomized_files[
                                                                                    number_of_training_samples:]
    return training_files, validation_files

# Reads the training/validation audio and concats it into a single array for the NN
def load_generic_audio(training_files, validation_files, sample_rate=16000):
    '''Generator that yields audio waveforms from the directory.'''

    # Concat training data
    training_data = []
    for training_filename in training_files:
        audio, _ = librosa.load(training_filename, sr=sample_rate, mono=True)
        audio = audio.reshape(-1, 1)
        training_data = training_data + audio.tolist()

    # Concat validation data
    validation_data = []
    for validation_filename in validation_files:
        audio, _ = librosa.load(validation_filename, sr=sample_rate, mono=True)
        audio = audio.reshape(-1, 1)
        validation_data = validation_data + audio.tolist()

    return np.array(training_data), np.array(validation_data)


def convertToFrames(audio, frame_size, frame_shift):
    X = []
    Y = []
    audio_len = len(audio)
    for i in range(0, audio_len - frame_size - 1, frame_shift):
        frame = audio[i:i + frame_size]
        if len(frame) < frame_size:
            break
        if i + frame_size >= audio_len:
            break
        temp = audio[i + frame_size]
        target_val = int((np.sign(temp) * (np.log(1 + 256 * abs(temp)) / (np.log(1 + 256))) + 1) / 2.0 * 255)
        X.append(frame.reshape(frame_size, 1))
        Y.append((np.eye(256)[target_val]))
    return np.array(X), np.array(Y)


def createDataset(audio_data, batch_size, frame_size, frame_shift):
    data_frames = convertToFrames(audio_data, frame_size, frame_shift)
    #print("data_frames: ", data_frames[0].shape)

    ds = tf.data.Dataset.from_tensor_slices(data_frames)

    ds = ds.repeat()

    ds = ds.batch(batch_size)
    #ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds




