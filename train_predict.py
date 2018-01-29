import sys
import json
import build_model
import data_helper
import numpy as np
import pandas as pd

def train_predict():
	"""Train and predict time series data"""

	# Load command line arguments 
	train_file = sys.argv[1]
	parameter_file = sys.argv[2]

	# Load training parameters
	params = json.loads(open(parameter_file).read())

	# Load time series dataset, and split it into train and test
	X_train, y_train, X_test, y_test = data_helper.load_timeseries(train_file, params['window_size'], True)

	# Build RNN (LSTM) model
	lstm_layer = [1, params['window_size'], params['hidden_unit'], 1]
	model = build_model.rnn_lstm(lstm_layer, params)

	# Train RNN (LSTM) model with time series train set
	model.fit(
		X_train,
		y_train,
		batch_size=params['batch_size'],
		epochs=params['epochs'],
		validation_split=params['validation_split'])

	# Predict next time stamp 
	predicted = build_model.predict_next_timestamp(model, X_test)        

	print(type(predicted))
	print(predicted.shape)
	for i in range(len(predicted)):
		print(predicted[i], y_test[i])

	data_helper.plot_results(predicted, y_test)
	print('training is done')

if __name__ == '__main__':
	# python3 train_predict.py ./data/sales.csv ./training_config.json_
	train_predict()
