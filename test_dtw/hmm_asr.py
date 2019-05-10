# Thu phan biet one, two

import sounddevice as sd
import soundfile as sf
from IPython.display import Audio
import numpy as np
import librosa
from dtw import dtw

def record_sound(duration, filename=None, fs=44100):
    sd.play(5.0*np.sin(2*np.pi*940*np.arange(fs)/fs), samplerate=fs, blocking=True)
    sd.play(0.0, samplerate=fs, blocking=True)
    print("recording")
    data = sd.rec(int(duration*fs), samplerate=fs, channels=1, blocking=True)
    print("done recording")
    if filename is not None:
        sf.write(filename, data, samplerate=fs)
    return data.T, fs

#test recording
# data, fs = record_sound(2, filename='Project/test.wav')
# Audio(data=data, rate=fs)

# n_sample = 10
# for i in range(n_sample):
#     record_sound(1, filename='Project/one_{}.wav'.format(i))
#
# for i in range(n_sample):
#     record_sound(1, filename='Project/two_{}.wav'.format(i))

def get_mfcc(filename):
    data, fs = librosa.load(filename, sr=None)
    mfcc = librosa.feature.mfcc(data, fs, hop_length=128, n_fft=1024)
    return mfcc

import copy
def preprocess_mfcc(mfcc):
    mfcc_cp = copy.deepcopy(mfcc)
    for i in range(mfcc.shape[1]):
        mfcc_cp[:,i] = mfcc[:,i] - np.mean(mfcc[:,i])
        mfcc_cp[:,i] = mfcc_cp[:,i]/np.max(np.abs(mfcc_cp[:,i]))
    return mfcc_cp

all_mfcc_one = [get_mfcc('one_{}.wav'.format(i)) for i in range(10)]
all_mfcc_two = [get_mfcc('two_{}.wav'.format(i)) for i in range(10)]

def dist(mfcc1, mfcc2):
	return dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: np.linalg.norm(x - y, ord=1))[0]

def one_or_two() :
	record_sound(duration=1, filename='test.wav')
	mfccTest = get_mfcc('test.wav')
	one = []
	print('one')
	for i in range(10):
		x = dist(all_mfcc_one[i],mfccTest)
		print(x)
		one.append(x)
	print('two')
	two = []
	for i in range(10):
		x = dist(all_mfcc_two[i],mfccTest)
		print(x)
		two.append(x)
	if sum(one)<sum(two):
		print('one')
	else:
		print('two')
	sd.play(5.0*np.sin(2*np.pi*940*np.arange(44100)/44100), samplerate=44100, blocking=True)

one_or_two()
