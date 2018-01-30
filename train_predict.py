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
	x_train, y_train, x_test, y_test, x_test_raw, y_test_raw, last_window_raw, last_window = data_helper.load_timeseries(train_file, params['window_size'])

	# Build RNN (LSTM) model
	lstm_layer = [1, params['window_size'], params['hidden_unit'], 1]
	model = build_model.rnn_lstm(lstm_layer, params)

	# Train RNN (LSTM) model with time series train set
	model.fit(
		x_train,
		y_train,
		batch_size=params['batch_size'],
		epochs=params['epochs'],
		validation_split=params['validation_split'])

	# Predict next time stamp 
	predicted = build_model.predict_next_timestamp(model, x_test)        

	for i in range(len(predicted)):
		print(predicted[i], y_test[i])

	print('raw')
	predicted_raw = []
	for i in range(len(x_test_raw)):
		predicted_raw.append((predicted[i] + 1) * x_test_raw[i][0])
		print(x_test_raw[i], y_test_raw[i])

	data_helper.plot_results(predicted, y_test, predicted_raw, y_test_raw)
	print('training is done')

	print('last')
	print(type(x_test))
	print(x_test[0])
	print(type(last_window))
	print(last_window)
	print(last_window_raw)

	next_timestamp = build_model.predict_next_timestamp(model, last_window)
	print(next_timestamp)
	print(type(next_timestamp))
	next_timestamp_raw = (next_timestamp[0] + 1) * last_window_raw[0][0]
	print(next_timestamp_raw)

if __name__ == '__main__':
	# python3 train_predict.py ./data/sales.csv ./training_config.json_
	train_predict()
