### Highlight:
- This is a **Time Series Forecasting** problem.
- The purpose of this project is to **Forecast next timestamp** given a sequence of history values.
- This module was built with **Recurrent Neural Network (RNN)** on top of **[Tensorflow](https://github.com/tensorflow/tensorflow)** and **[Keras](https://github.com/keras-team/keras)**.

### Why apply RNN (LSTM) on time series datasets?
> The expression long short-term refers to the fact that LSTM is a model for the **short-term memory which can last for a long period of time**. An LSTM is well-suited to classify, process and **predict time series given time lags of unknown size and duration between important events**.
- [Long short-term memory](https://en.wikipedia.org/wiki/Long_short-term_memory)

> Recurrent neural networks are a type of neural network that **add the explicit handling of order in input observations**.

> This capability suggests that the promise of recurrent neural networks is to **learn the temporal context of input sequences in order to make better predictions**. That is, that the suite of lagged observations required to make a prediction no longer must be diagnosed and specified as in traditional time series forecasting, or even forecasting with classical neural networks. Instead, the temporal dependence can be learned, and perhaps changes to this dependence can also be learned.

- [The Promise of Recurrent Neural Networks for Time Series Forecasting](https://machinelearningmastery.com/promise-recurrent-neural-networks-time-series-forecasting/)

### Data:
- Input: **a sequence of history values**
  - 2017-01-01,339.7
  - 2017-02-01,440.4
  - 2017-03-01,315.9
  - 2017-04-01,439.3
  - 2017-05-01,401.3
  - 2017-06-01,437.4
  - 2017-07-01,575.5
  - 2017-08-01,407.6
  - 2017-09-01,682.0
  - 2017-10-01,475.3
  - 2017-11-01,581.3
  - 2017-12-01,646.9

- Output: **the value on next timestamp**
  - 2018-01-01,678.5

### Train & Predict:
- Example 1: ```python3 train_predict.py ./data/sales.csv ./training_config.json```
- Example 2: ```python3 train_predict.py ./data/daily-minimum-temperatures-in-me.csv ./training_config.json ```

### Reference:
- [Time Series Forecasting with the Long Short-Term Memory Network in Python](https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/)
- [LSTM Neural Network for Time Series Prediction](http://www.jakob-aungiers.com/articles/a/LSTM-Neural-Network-for-Time-Series-Prediction)
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
