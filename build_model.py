import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

def rnn_lstm(layers, params):
	"""Build RNN (LSTM) model on top of Keras and Tensorflow"""

	model = Sequential()
	model.add(LSTM(input_shape=(layers[1], layers[0]), output_dim=layers[1], return_sequences=True))
	model.add(Dropout(params['dropout_keep_prob']))
	model.add(LSTM(layers[2], return_sequences=False))
	model.add(Dropout(params['dropout_keep_prob']))
	model.add(Dense(output_dim=layers[3]))
	model.add(Activation("tanh"))

	model.compile(loss="mean_squared_error", optimizer="rmsprop")
	return model

def predict_next_timestamp(model, history):
	"""Predict the next time stamp given a sequence of history data"""

	prediction = model.predict(history)
	prediction = np.reshape(prediction, (prediction.size,))
	return prediction 

