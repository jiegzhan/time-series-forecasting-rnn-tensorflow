import time
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

def rnn_lstm(layers):
	model = Sequential()

	model.add(LSTM(
		input_shape=(layers[1], layers[0]),
		output_dim=layers[1],
		return_sequences=True))
	model.add(Dropout(0.2))

	model.add(LSTM(
		layers[2],
		return_sequences=False))
	model.add(Dropout(0.2))

	model.add(Dense(
		output_dim=layers[3]))
	model.add(Activation("linear"))

	start = time.time()
	model.compile(loss="mse", optimizer="rmsprop")
	print("> Compilation Time : ", time.time() - start)
	return model

