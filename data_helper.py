import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_timeseries(filename, window_size):
	"""Load time series dataset"""

	series = pd.read_csv(filename, sep=',', header=0, index_col=0, squeeze=True)
	data = series.values

	sequence_length = window_size + 1

	raw = []
	for index in range(len(data) - sequence_length):
		raw.append(data[index: index + sequence_length])

	result = normalise_windows(raw)

	raw = np.array(raw)
	result = np.array(result)

	split_ratio = round(0.8 * result.shape[0])
	train = result[:int(split_ratio), :]
	np.random.shuffle(train)

	x_train = train[:, :-1]
	y_train = train[:, -1]

	x_test = result[int(split_ratio):, :-1]
	y_test = result[int(split_ratio):, -1]

	x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
	x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  

	x_test_raw = raw[int(split_ratio):, :-1]
	y_test_raw = raw[int(split_ratio):, -1]

	# Last window, for next time stamp prediction
	last_raw = [data[-window_size:]]
	last = normalise_windows(last_raw)
	last = np.array(last)
	last = np.reshape(last, (last.shape[0], last.shape[1], 1))

	return [x_train, y_train, x_test, y_test, x_test_raw, y_test_raw, last_raw, last]

def normalise_windows(window_data):
	"""Normalize data"""

	normalised_data = []
	for window in window_data:
		normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
		normalised_data.append(normalised_window)
	return normalised_data

def plot_results(predicted_data, true_data, predicted_raw, true_data_raw):
	"""Plot predictions VS true data"""

	fig = plt.figure(facecolor='white')
	ax = fig.add_subplot(221)
	ax.plot(true_data, label='True Data')
	plt.plot(predicted_data, label='Prediction')
	plt.legend()

	plt.subplot(223)
	plt.plot(true_data_raw, label='true_data_raw')
	plt.plot(predicted_raw, label='predicted_raw')	
	plt.legend()
	plt.show()
