import numpy as np
from pydub import AudioSegment
import random
from utils import *


Tx = 5511 		# The number of time steps input to the model from the spectrogram
n_freq = 101 	# Number of frequencies input to the model at each time step of the spectrogram
Ty = 1375 		# The number of time steps in the output of the model

positives, negatives, backgrounds = load_raw_audio()

def get_random_segment_time(segment_length):
	start = np.random.randint(low=0, high=10000-segment_length)  # 10000ms of background
	end = start + segment_length - 1
	return (start, end)

def is_overlapping(segment, previous_segments):
	start, end = segment
	for previous_start, previous_end in previous_segments:
		if start <= previous_end and end >= previous_start:
			return True
	return False

def insert_audio(background, audio, previous_segments):
	segment_length = len(audio)
	segment = get_random_segment_time(segment_length)
	while is_overlapping(segment, previous_segments):
		segment = get_random_segment_time(segment_length)
	previous_segments.append(segment)
	new_background = background.overlay(audio, position=segment[0])
	return new_background, segment

def insert_positive_label(y, segment_end):
	# labels of the 50 output steps after the end segment will be set to 1
	segment_end_y = int(segment_end * Ty / 10000.0)
	for i in range(segment_end_y + 1, segment_end_y + 51):
		if i < Ty:
			y[0,i] = 1
	return y

def create_training_sample(file_name, backgrounds, positives, negatives):
	background = backgrounds[np.random.randint(0,2)]
	background = background - 20
	y = np.zeros((1,Ty))
	previous_segments = []

	# insert negatives randomly
	number_of_negatives = np.random.randint(1,3)
	random_indexes = np.random.randint(len(negatives), size=number_of_negatives)
	random_negatives = [negatives[i] for i in random_indexes]

	for negative in random_negatives:
		# break
		background, _ = insert_audio(background, negative, previous_segments)

	#insert positives randomly
	number_of_positives = np.random.randint(1,3)
	random_indexes = np.random.randint(len(positives), size=number_of_positives)
	random_positives = [positives[i] for i in random_indexes]

	for positive in random_positives:
		background, segment = insert_audio(background, positive, previous_segments)
		_, segment_end = segment
		y = insert_positive_label(y, segment_end)

	background = match_target_amplitude(background, -20.0)
	background.export(file_name, format="wav")
	x = graph_spectrogram(file_name)
	return x, y

if __name__=="__main__":
	train_set_size = 1000
	X = np.zeros((train_set_size, Tx, n_freq))
	y = np.zeros((train_set_size, Ty, 1))
	for i in range(train_set_size):
		print("{}/{} Creating...".format(i+1,train_set_size))
		new_X, new_y = create_training_sample("./train_audio/train_{}.wav".format(i), backgrounds, positives, negatives)
		X[i] = new_X.swapaxes(0,1)
		y[i] = new_y.swapaxes(0,1)

	print("Writing...")
	np.save("./train_set/X_train.npy", X)
	np.save("./train_set/Y_train.npy", y)
