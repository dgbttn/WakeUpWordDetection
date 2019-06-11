import numpy as np

from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam


Tx = 5511 		# The number of time steps input to the model from the spectrogram
n_freq = 101 	# Number of frequencies input to the model at each time step of the spectrogram
Ty = 1375 		# The number of time steps in the output of the model

def TrainModel(input_shape):

    X_input = Input(shape = input_shape)

    # Step 1: CONV layer (≈4 lines)
    X = Conv1D(196, kernel_size=15, strides=4)(X_input)		# CONV1D
    X = BatchNormalization()(X)                				# Batch normalization
    X = Activation('relu')(X)                     			# ReLu activation
    X = Dropout(0.8)(X)                        				# dropout (use 0.8)

    # Step 2: First GRU Layer (≈4 lines)
    X = GRU(units = 128, return_sequences = True)(X) 		# GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)                                 	# dropout (use 0.8)
    X = BatchNormalization()(X)                    			# Batch normalization

    # Step 3: Second GRU Layer (≈4 lines)
    X = GRU(units = 128, return_sequences = True)(X)   		# GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)                                		# dropout (use 0.8)
    X = BatchNormalization()(X)                    			# Batch normalization
    X = Dropout(0.8)(X)                                  	# dropout (use 0.8)

    # Step 4: Time-distributed dense layer (≈1 line)
    X = TimeDistributed(Dense(1, activation = "sigmoid"))(X) # time distributed  (sigmoid)

    model = Model(inputs = X_input, outputs = X)
    return model


if __name__=="__main__":
	print("Loading data...")
	X = np.load("./train_set/X_train.npy")
	y = np.load("./train_set/Y_train.npy")

	print("Creating model...")
	model = TrainModel(input_shape = (Tx, n_freq))
	opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
	print("Training model...")
	model.fit(X, y, batch_size=5, epochs=1)

	print("Saving model...")
	model.save("./my_model.h5")
