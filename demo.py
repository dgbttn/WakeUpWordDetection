import numpy as np
from pydub import AudioSegment
import IPython
import matplotlib.mlab as mlab
from scipy.io.wavfile import write
from keras.models import load_model

# LOAD A PRE-TRAIN MODEL

print("Loading model...")
model = load_model('./my_model.h5')
print("Loading done")

# Detect trigger word functions
def detect_triggerword_spectrum(x):
    x  = x.swapaxes(0,1)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)
    return predictions.reshape(-1)

def has_new_triggerword(predictions, chunk_duration, feed_duration, threshold=0.5):
    predictions = predictions > threshold
    chunk_predictions_samples = int(len(predictions) * chunk_duration / feed_duration)
    chunk_predictions = predictions[-chunk_predictions_samples:]
    level = chunk_predictions[0]
    for pred in chunk_predictions:
        if pred > level:
            return True
        else:
            level = pred
    return False


# RECORD AUDIO STREAM FROM MIC

chunk_duration = 0.5
fs = 44100
chunk_samples = int(fs * chunk_duration)

feed_duration = 10
feed_samples = int(fs * feed_duration)

assert feed_duration/chunk_duration == int(feed_duration/chunk_duration)

def get_spectrogram(data):
    nfft = 200
    fs = 8000
    noverlap = 120
    nchannels = data.ndim
    if nchannels == 1:
        pxx, _, _ = mlab.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, _, _ = mlab.specgram(data[:,0], nfft, fs, noverlap = noverlap)
    return pxx


# Audio Stream

def get_audio_input_stream(callback):
    stream = pyaudio.PyAudio().open(
        format=pyaudio.paInt16,
        channels=1,
        rate=fs,
        input=True,
        frames_per_buffer=chunk_samples,
        input_device_index=0,
        stream_callback=callback)
    return stream


import pyaudio
from queue import Queue
from threading import Thread
import time

import os
# from dotenv import load_dotenv
# load_dotenv()

# Queue to communiate between the audio callback and main thread
q = Queue()
run = True
silence_threshold = 100

# Run the demo for a timeout seconds
timeout = time.time() + 0.5*60  # 0.5 minutes from now

# Data buffer for the input wavform
data = np.zeros(feed_samples, dtype='int16')

def callback(in_data, frame_count, time_info, status):
	global run, timeout, data, silence_threshold, feed_samples
	# if time.time() > timeout:
	# 	run = False
	data0 = np.frombuffer(in_data, dtype='int16')
	if np.abs(data0).mean() < silence_threshold:
		print('-',end = '')
		return (in_data, pyaudio.paContinue)
	else:
		print('.',end = '')
	data = np.append(data,data0)

	if len(data) > feed_samples:
		data = data[-feed_samples:]
        # Process data async by sending a queue.
		q.put(data)
	return (in_data, pyaudio.paContinue)

if __name__ == "__main__":

	try:
		print("Starting...")
		stream = get_audio_input_stream(callback)
		stream.start_stream()
		print("Stream opened")

		while run:
			if (q.empty()):
				continue
			data = q.get()
			spectrum = get_spectrogram(data)
			preds = detect_triggerword_spectrum(spectrum)
			new_trigger = has_new_triggerword(preds, chunk_duration, feed_duration)
			print("")
			if new_trigger:
				print('1',end = '')
				os.system("C:/Users/dgbttn/AppData/Local/CocCoc/Browser/Application/browser.exe")

	except (KeyboardInterrupt, SystemExit):
	    stream.stop_stream()
	    stream.close()
	    timeout = time.time()
	    run = False

	stream.stop_stream()
	stream.close()

	print("Finish!")
