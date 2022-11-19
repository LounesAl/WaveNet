import tqdm 
import numpy as np
from pathlib import Path
from scipy.io.wavfile import write
from tensorflow import keras
from config import configuration
from hyperparameters import hyperparameters
from audio_preprocessing import (load_train_valid_filenames, load_generic_audio)


LJ_DIRECTORY = Path('./data_wav/')  # Dataset Directory
LOG_DIRECTORY = Path('./saved_data/model_logs/')



# Generates an audio clip from the NN. After each sample is collected, the inverse of the softmax is taken to normalize the sound
def get_audio_from_model(model, sr, duration, seed_audio, frame_size):
    new_audio = np.zeros((sr * duration))
    for curr_sample_idx in tqdm.tqdm(range(new_audio.shape[0])):
        distribution = np.array(model.predict(seed_audio.reshape(1, frame_size, 1)), dtype=float).reshape(256)
        distribution /= distribution.sum().astype(float)
        predicted_val = np.random.choice(range(256), p=distribution)
        ampl_val_8 = predicted_val / 255.0
        ampl_val_16 = (np.sign(ampl_val_8) * (1/255.0) * ((1 + 256.0)**abs(
            ampl_val_8) - 1)) * 2**15
        new_audio[curr_sample_idx] = ampl_val_16
        seed_audio[:-1] = seed_audio[1:]
        seed_audio[-1] = ampl_val_16
    return new_audio.astype(np.int16)


def generateAudioFromModel(model, model_id, sr=16000, frame_size=256,
                            num_files=1, generated_seconds=1, validation_audio=None):
    audio_context = validation_audio[:frame_size]

    for i in range(num_files):
        new_audio = get_audio_from_model(model, sr, generated_seconds, audio_context, frame_size)
        audio_context = validation_audio[i:i + frame_size]
        log_dir = Path(LOG_DIRECTORY / model_id)
        wavname = (model_id + "_sample_" + str(i) + '.wav')
        outputPath = "saved_data/"+ wavname
        print("Saving File", outputPath)
        write(outputPath, sr, new_audio)



def generate(path_model, model_id, path_validation_audio):

    configuration()
    try:
        model = keras.models.load_model(path_model)
    except :
        pass

    training_files, validation_files = load_train_valid_filenames(  LJ_DIRECTORY, hyperparameters['num_samples'],
                                                                                        percent_training=0.9    )
    _, validation_audio = load_generic_audio(training_files, validation_files, sample_rate=hyperparameters["sample_rate"])

    print("Generating Audio.")
    generateAudioFromModel(model, model_id, sr=hyperparameters["sample_rate"], frame_size=hyperparameters["frame_size"],
                           num_files=1, generated_seconds=1, validation_audio=validation_audio)
    print("Program Complete.")


if __name__ == '__main__':
    generate()