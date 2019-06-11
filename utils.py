import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
from pydub import AudioSegment

def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)
    return rate, data

# Calculate and plot spectrogram for a wav audio file
def graph_spectrogram(wav_file):
    rate, data = get_wav_info(wav_file)
    nfft = 200 		# Length of each window segment
    fs = 8000		# Sampling frequencies
    noverlap = 120 	# Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
    return pxx[:,:5511]

# Standardize volume of audio
def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

def load_raw_audio():
    positives = []
    backgrounds = []
    negatives = []

    for filename in os.listdir("./raw_data/positives"):
        if filename.endswith("wav"):
            positive = AudioSegment.from_wav("./raw_data/positives/"+filename)
            positives.append(positive)

    for filename in os.listdir("./raw_data/backgrounds"):
        if filename.endswith("wav"):
            background = AudioSegment.from_wav("./raw_data/backgrounds/"+filename)
            backgrounds.append(background)

    for filename in os.listdir("./raw_data/negatives"):
        if filename.endswith("wav"):
            negative = AudioSegment.from_wav("./raw_data/negatives/"+filename)
            negatives.append(negative)

    return positives, negatives, backgrounds
