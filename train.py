import sys
import data_helper
import pandas as pd
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy
import lstm
import time
import matplotlib.pyplot as plt
import build_model

# scale train and test data to [-1, 1]
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = numpy.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
	return model

# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]

def train():

	# Load dataset
	train_file = sys.argv[1]
	series = read_csv(train_file, sep='\t', header=0, index_col=0, squeeze=True)
	vals = series.values
	print(vals)
	print(type(vals))
	print(vals[0])
	print(type(vals[0]))

	# transform data to be stationary
	raw_values = series.values
	diff_values = data_helper.difference(raw_values, 1)
	print('this is diff_values')
	print(diff_values)

	# transform data to be supervised learning
	supervised = data_helper.timeseries_to_supervised(diff_values, 1)
	supervised_values = supervised.values

	# split data into train and test-sets
	train, test = supervised_values[0:-12], supervised_values[-12:]

	# transform the scale of the data
	scaler, train_scaled, test_scaled = scale(train, test)

	# fit the model
	lstm_model = fit_lstm(train_scaled, 1, 1000, 4)
	# forecast the entire training dataset to build up state for forecasting
	train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
	lstm_model.predict(train_reshaped, batch_size=1)

	# walk-forward validation on the test data
	predictions = list()
	for i in range(len(test_scaled)):
		# make one-step forecast
		X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
		yhat = forecast_lstm(lstm_model, 1, X)
		# invert scaling
		yhat = invert_scale(scaler, X, yhat)
		# invert differencing
		yhat = data_helper.inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
		# store forecast
		predictions.append(yhat)
		expected = raw_values[len(train) + i + 1]
		print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))

	# report performance
	rmse = sqrt(mean_squared_error(raw_values[-12:], predictions))
	print('Test RMSE: %.3f' % rmse)
	# line plot of observed vs predicted
	# pyplot.plot(raw_values[-12:])
	# pyplot.plot(predictions)
	# pyplot.show()

def train_lstm():
	global_start_time = time.time()
	epochs  = 50
	seq_len = 6

	print('> Loading data... ')

	# Load time series dataset
	train_file = sys.argv[1]

	# X_train, y_train, X_test, y_test = lstm.load_data('sp500.csv', seq_len, True)
	X_train, y_train, X_test, y_test = data_helper.load_timeseries(train_file, seq_len, True)

	print('> Data Loaded. Compiling...')

	# model = lstm.build_model([1, 6, 100, 1])
	model = build_model.rnn_lstm([1, 6, 100, 1])

	model.fit(X_train, y_train, batch_size=2, nb_epoch=epochs, validation_split=0.05)

	#predictions = lstm.predict_sequences_multiple(model, X_test, seq_len, 50)
	#predicted = lstm.predict_sequence_full(model, X_test, seq_len)
	predicted = lstm.predict_point_by_point(model, X_test)        

	print(type(predicted))
	print(predicted.shape)
	for i in range(len(predicted)):
		print(predicted[i], y_test[i])

	print('Training duration (s) : ', time.time() - global_start_time)
	# plot_results_multiple(predictions, y_test, 50)
	data_helper.plot_results(predicted, y_test)
	print('training is done')

if __name__ == '__main__':
	train_lstm()
