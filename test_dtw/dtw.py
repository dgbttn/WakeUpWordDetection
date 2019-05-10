import librosa
from dtw import dtw
import IPython.display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

def show_image(dist,cost,path):
    plt.imshow(cost.T, origin='lower', interpolation='nearest')
    plt.plot(path[0], path[1], 'w')
    plt.xlim((-0.5, cost.shape[0]-0.5))
    plt.ylim((-0.5, cost.shape[1]-0.5))

import copy
def preprocess_mfcc(mfcc):
    mfcc_cp = copy.deepcopy(mfcc)
    for i in range(mfcc.shape[1]):
        mfcc_cp[:,i] = mfcc[:,i] - np.mean(mfcc[:,i])
        mfcc_cp[:,i] = mfcc_cp[:,i]/np.max(np.abs(mfcc_cp[:,i]))
    return mfcc_cp

y1, sr1 = librosa.load('elias_mothers_milk_word.wav')
y2, sr2 = librosa.load('chris_mothers_milk_word.wav')
y3, sr3 = librosa.load('yaoquan_mothers_milk_word.wav')
yX, srX = librosa.load('chris_mothers_milk_sentence_fast.wav')

mfcc1 = librosa.feature.mfcc(y1, sr1)
mfcc2 = librosa.feature.mfcc(y2, sr2)
mfcc3 = librosa.feature.mfcc(y3, sr3)
mfccX = librosa.feature.mfcc(yX, srX)

IPython.display.Audio(data=yX, rate=sr1)

mfcc1 = preprocess_mfcc(mfcc1)
mfcc2 = preprocess_mfcc(mfcc2)
mfcc3 = preprocess_mfcc(mfcc3)
mfccX = preprocess_mfcc(mfccX)

window_size = (mfcc1.shape[1]+mfcc2.shape[1]+mfcc3.shape[1])//3//2
dists = np.zeros(mfccX.shape[1] - window_size)

for i in range(len(dists)):

    mfcci = mfccX[:,i:i+window_size]
    #mfcci = mfcci/np.max(mfcci)

    dist1i = dtw(mfcc1.T, mfcci.T, dist = lambda x, y: np.exp(np.linalg.norm(x - y, ord=1)))[0]
    dist2i = dtw(mfcc2.T, mfcci.T, dist = lambda x, y: np.exp(np.linalg.norm(x - y, ord=1)))[0]
    dist3i = dtw(mfcc3.T, mfcci.T, dist = lambda x, y: np.exp(np.linalg.norm(x - y, ord=1)))[0]
    dists[i] = (dist1i + dist2i + dist3i)/3

plt.plot(dists)

# select minimum distance window
word_match_idx = dists.argmin()
word_match_idx
word_match_idx_bnds = np.array([word_match_idx,np.ceil(word_match_idx+window_size)], dtype=np.int)
samples_per_mfcc = 512#len(yX)/mfccX.shape[1]
word_samp_bounds = (2//2) + (word_match_idx_bnds*samples_per_mfcc)

word = yX[word_samp_bounds[0]:word_samp_bounds[1]]

plt.plot(word)


IPython.display.Audio(data=word, rate=sr1)
