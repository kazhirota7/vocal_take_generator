import os

import numpy as np

from AE import VAE


LEARNING_RATE = 0.0002
BATCH_SIZE = 16
EPOCHS = 150

SPECTROGRAMS_PATH = os.getcwd() + "/spectro"


def load_fsdd(spectrograms_path):
    x_train = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path) # (n_bins, n_frames, 1)
            x_train.append(spectrogram)
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis] # -> (4, 256, 563, 1)
    return x_train


def train(x_train, learning_rate, batch_size, epochs):
    # autoencoder = VAE(
    #     input_shape=(256, 512, 1),
    #     conv_filters=(512, 256, 128, 64, 32),
    #     conv_kernels=(3, 3, 3, 3, 3),
    #     conv_strides=(2, 2, 2, 2, (2, 1)),
    #     latent_space_dim=128
    # )
    autoencoder = VAE(
        input_shape=(256, 512, 1),
        conv_filters=(32, 64, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3, 3),
        conv_strides=(1, 2, 2, 2, (2, 1)),
        latent_space_dim = 128
    )
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    autoencoder.train(x_train, batch_size, epochs)
    return autoencoder


if __name__ == "__main__":
    x_train = load_fsdd(SPECTROGRAMS_PATH)
    autoencoder = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    autoencoder.save("model")