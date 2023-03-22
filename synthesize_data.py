from scipy import signal
from scipy.io.wavfile import write
import librosa
import os
import numpy as np
import yaml

SAMPLE_RATE = 41000

with open("test.yaml", 'r') as stream:
    parameters = yaml.load(stream, Loader=yaml.FullLoader)

#get number of sound events to place in file
SOUND_EVENT_NUM = np.random.randint(low=parameters["number_of_sound_events"][0], high=parameters["number_of_sound_events"][1])
background_type = np.random.choice(parameters["backgrounds"], 1, p=parameters["backgrounds_pv"])[0]
print(background_type)
possible_backgrounds = [i for i in os.listdir(parameters["background_db_location"]) if 
                        os.path.isfile(os.path.join(parameters["background_db_location"],i)) and background_type in i]
background = np.random.choice(possible_backgrounds, 1)[0]
print(background)

#select
for sounds in range(SOUND_EVENT_NUM):

    print(parameters["hrir_database"])
    hrir = np.random.choice(parameters["hrir_database"], 1, p=parameters["hrir_database_pv"])
    print(hrir)
    angle = np.random.choice([*range(len(parameters["hrir_angle_range"]))], 1, p=parameters["hrir_angle_range_pv"])
    print(parameters["hrir_angle_range"][angle[0]])
    s_class = np.random.choice(parameters["sound_event_class"], 1, p=parameters["sound_event_class_pv"])
    print(s_class)
exit()

formats = ["44K_16bit", "48K_24bit", "96K_24bit"]

input_file = "test.aiff"

hrir_database = []
hrir_database_pv = []
hrir_angle_range = []
hrir_angle_range_pv = []
sound_event_class = []
sound_event_class_pv = []

azimuth = 180
elevation = 0

def load_hrir(az, ele, format):
    filename = "azi_" + str(az) + ",0_" + "ele_" + str(ele) + ",0.wav"
    fullpath = "HRIRs/D1_HRIR_WAV/" + format + "/" + filename
    audio = librosa.load(fullpath, mono=False, sr=SAMPLE_RATE)
    return audio

# videoclip = moviepy.editor.VideoFileClip("Videos/shopping_mall-london-256-7704.mp4")
# audioclip = moviepy.editor.AudioFileClip("Audio/shopping_mall-london-256-7704.wav")

# videoclip.set_audio(audioclip)

# videoclip.write_videofile("test.mp4")

hrir,_ = load_hrir(azimuth,elevation,formats[0])
audio,_ = librosa.load("test.aiff", sr=SAMPLE_RATE)
background,_ = librosa.load("Audio/shopping_mall-london-256-7704.wav", sr=SAMPLE_RATE, mono=False)
background = background
audio = np.pad(audio, (0, len(background[0])-len(audio)))

left = signal.convolve(audio, hrir[0], mode='same')
right = signal.convolve(audio, hrir[1], mode='same')

audio = np.array([left, right])

print(audio.shape)
print(background.shape)

audio += background


output = audio.T

write("output.wav", rate=SAMPLE_RATE, data=output)
