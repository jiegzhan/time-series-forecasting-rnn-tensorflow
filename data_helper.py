import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_timeseries(filename, seq_len, normalise_window):
	series = pd.read_csv(filename, sep='\t', header=0, index_col=0, squeeze=True)
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
	normalised_data = []
	for window in window_data:
		normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
		normalised_data.append(normalised_window)
	return normalised_data

def plot_results(predicted_data, true_data):
	fig = plt.figure(facecolor='white')
	ax = fig.add_subplot(111)
	ax.plot(true_data, label='True Data')
	plt.plot(predicted_data, label='Prediction')
	plt.legend()
	plt.show()

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return pd.Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = pd.DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = pd.concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df
