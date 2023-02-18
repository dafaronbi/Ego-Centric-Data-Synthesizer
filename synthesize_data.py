from scipy import signal
from scipy.io.wavfile import write
import librosa
import os
import numpy as np

SAMPLE_RATE = 44000

formats = ["44K_16bit", "48K_24bit", "96K_24bit"]

input_file = "test.aiff"

azimuth = 100
elevation = 15

def load_hrir(az, ele, format):
    filename = "azi_" + str(az) + ",0_" + "ele_" + str(ele) + ",0.wav"
    fullpath = "SADIE_D1_HRIR_WAV/" + format + "/" + filename
    audio = librosa.load(fullpath, mono=False, sr=SAMPLE_RATE)
    return audio

hrir,_ = load_hrir(azimuth,elevation,formats[0])
audio,_ = librosa.load("test.aiff", sr=SAMPLE_RATE)

left = signal.convolve(audio, hrir[0])
right = signal.convolve(audio, hrir[1])

output = np.array([left, right]).T

write("output.wav", rate=SAMPLE_RATE, data=output)
