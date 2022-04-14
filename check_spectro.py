from load_audio import *
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write


# num_samples = 4
# dataset = []
# sample_rate = 48000
# duration = 3  # seconds
#
# for i in range(num_samples):
#     audio_array = audio_to_array(os.getcwd()+"/audio/audio" + str(i+1) + ".m4a")
#     dataset.append(audio_array[0:sample_rate*duration])
# dataset = np.array(dataset)
#
# write("test.wav", 48000, dataset[0])
test = np.load('old/audio1.m4a.npy')
test2 = np.load('old/audio2.m4a.npy')
print(test.shape)
print(test2.shape)
test3 = np.load('old/audio3.m4a.npy')
test4 = np.load('old/audio4.m4a.npy')
print(test3.shape)
print(test4.shape)