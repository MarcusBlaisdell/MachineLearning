
'''
Date: 04/13/2019

Input: Microsoft daily stock price history => training and testing data
The program: Use the first a% closing prices to predict the last (100-a)% closing prices
ML method: LSTM
Output: a graph comparing the true prices and predicted prices
        and performance mesures

Credit: The base of this program is taken from
https://www.analyticsvidhya.com/blog/2018/10/predicting-stock-price-machine-learningnd-deep-learning-techniques-python/
'''

#importing required libraries
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
#from google.colab import files #read files from local machine
import io


#import packages
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error

#to plot within notebook
import matplotlib.pyplot as plt

#setting figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10

#for normalizing data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

def plot_history(history):
  hist = pd.DataFrame(history.history)
  print(hist)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.plot(hist['loss'])
  '''
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [close]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.ylim([0,1500000])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$close^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.ylim([0,200000000])
  plt.legend()
  '''
  plt.show()

#open a file (dataset)
#uploaded = files.upload()

#get filename
#for fn in uploaded.keys():
#  filename = fn

#Read the file, ask pandas to recognize the dates, set 'date' as index column
df = pd.read_csv("Microsoft.csv")
#df = pd.read_csv("MBIO.csv")
'''
column_names = ['date', 'close', 'volume', 'open', 'high', 'low']

df = pd.read_csv ("Microsoft.csv",
                            names=column_names, na_values = "?",
                            comment='\t', sep=",", skipinitialspace=True)
'''
#creating dataframe

df = df.sort_index(ascending=True, axis=0)

new_data = pd.DataFrame(index=range(0,len(df)),columns=['date', 'close'])

for i in range(0,len(df)):
    new_data['date'][i] = df['date'][i]
    new_data['close'][i] = df['close'][i]

#re-order the data in increment of time
new_data = new_data.sort_values('date')
new_data = new_data.reset_index(drop=True)


#setting index
new_data.index = new_data.date
new_data.drop('date', axis=1, inplace=True)


#creating train and test sets
dataset = new_data.values

train = dataset[0:2000,:]
valid = dataset[2000:,:]

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))


# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(x_train, y_train, epochs=3, batch_size=1, verbose=2)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
plot_history(history)


#predicting 246 values, using past 60 from the train data
inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)




train = new_data[:2000]
valid = new_data[2000:]

#MSE
mse = mean_squared_error(valid[['close']], closing_price)
print("Mean Squared Error = " + str(mse))


#for plotting
valid['Predictions'] = closing_price

plt.plot(train['close'])
plt.plot(valid[['close','Predictions']])
plt.show()
