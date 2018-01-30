import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_timeseries(filename, seq_len, normalise_window):
	"""Load time series dataset"""

	series = pd.read_csv(filename, sep=',', header=0, index_col=0, squeeze=True)
	data = series.values

	sequence_length = seq_len + 1
	result = []
	for index in range(len(data) - sequence_length):
		result.append(data[index: index + sequence_length])

	if normalise_window:
		result = normalise_windows(result)

	result = np.array(result)

	row = round(0.8 * result.shape[0])
	train = result[:int(row), :]
	np.random.shuffle(train)
	x_train = train[:, :-1]
	y_train = train[:, -1]
	x_test = result[int(row):, :-1]
	y_test = result[int(row):, -1]

	x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
	x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  

	return [x_train, y_train, x_test, y_test]

def normalise_windows(window_data):
	"""Normalize data"""

	normalised_data = []
	for window in window_data:
		normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
		normalised_data.append(normalised_window)
	return normalised_data

def plot_results(predicted_data, true_data):
	"""Plot predictions VS true data"""

	fig = plt.figure(facecolor='white')
	ax = fig.add_subplot(111)
	ax.plot(true_data, label='True Data')
	plt.plot(predicted_data, label='Prediction')
	plt.legend()
	plt.show()
