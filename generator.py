import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
from scipy.fft import fft, ifft
from pydub import AudioSegment

audio = AudioSegment.from_file('audio/example1.m4a')


print(audio)

