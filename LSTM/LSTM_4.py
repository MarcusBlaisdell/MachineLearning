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

#importing required libraries and pạckages
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt #to plot within notebook
from matplotlib.pylab import rcParams #setting figure size
from sklearn.preprocessing import MinMaxScaler #for normalizing data

#inital set up
rcParams['figure.figsize'] = 20,10
SCALER = MinMaxScaler(feature_range=(0, 1))
N_DATES_TO_PREDICT = 20


#Most hyper-parameters are set here as global variables
#Adjust values here for hyper-parameters tuning
TRAIN_DATA_PERCENTAGE = 0.8 # [0,1]
RANGE = 100 #use RANGE consecutive data points to predict the next data point
N_EPOCHS = 3
#Other hyper_parameters are in the function build_model()



#This function reads an input file from your local machine and returns the data
#The return data: has 'date' as the index column and 'close' as the only data column
#                 is in increasing order of dates
#                 is in pandas.DataFrame form
def get_data():

  #Read the file
  #df = pd.read_csv(io.BytesIO(uploaded[filename]))
  df = pd.read_csv("MBIO.csv")

  #creating dataframe with two columns and an index column
  new_data = pd.DataFrame(index=range(0,len(df)),columns=['date', 'close'])

  #transfer data from df to the new dataframe
  for i in range(0,len(df)):
      new_data['date'][i] = df['date'][i]
      new_data['close'][i] = df['close'][i]

  #re-order the data in increment of time
  new_data = new_data.sort_values('date')

  #setting index to 'date'
  new_data.index = new_data.date
  new_data.drop('date', axis=1, inplace=True)

  return new_data



#This function splits the data into X_train,y_train and X_test,y_test sets based on TRAIN_DATA_PERCENTAGE and RANGE
#Input data: type = numpy.ndarray, exp: [[134.1234] [243524.5] .... [24.576]]
#Return the normalized X_train, y_train and X_test, y_test sets of type datasetnumpy.ndarray
def split_train_test(data):
  #where to split train and test?
  index_to_split = int(TRAIN_DATA_PERCENTAGE * len(data))

  #normalize the dataset
  scaled_data = SCALER.fit_transform(data)

  X_train, y_train, X_test, y_test = [], [], [], []

  #get X_train, y_train
  for i in range(RANGE,index_to_split):
      X_train.append(scaled_data[i-RANGE:i,0])
      y_train.append(scaled_data[i,0])


  #prepare to get X_test, y_test
  #predicting y_test using past 60 from the train data
  inputs = data[index_to_split - RANGE:]
  inputs = inputs.reshape(-1,1)
  inputs  = SCALER.transform(inputs)

  #get X_test, y_test
  for i in range(RANGE,inputs.shape[0]):
      X_test.append(inputs[i-RANGE:i,0])
      y_test.append(inputs[i,0])

  #convert to numpy.ndarray to easily process
  X_train, y_train = np.array(X_train), np.array(y_train)
  X_test, y_test = np.array(X_test), np.array(y_test)

  #reshape X_train to prepare to train
  X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
  X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

  return X_train, y_train, X_test[0:1], y_test



#This function builds a LSTM model and returns it
#This function heavily uses global variables
def build_model():
  model = Sequential()
  model.add(LSTM(units=50, return_sequences=True, input_shape=(RANGE,1), stateful=True,batch_input_shape=(1,RANGE,1)))
  model.add(LSTM(units=50, stateful=True))
  model.add(Dense(1))
  model.compile(loss='mean_squared_error', optimizer='adam')

  return model



def main():
  #read the dataset
  new_data = get_data()

  #split data to train and test sets
  x_train, y_train, X_test, y_test = split_train_test(new_data.values)
  #repeat the same experiment for 'repeat' times to obtain more accurate results
  repeat = 1


  ############################
  #plt.figure()
  for i in range(repeat):
    #create and fit the LSTM network
    model = build_model()

    train_mse, test_mse = [], []
    prediction = []
    for j in range(N_EPOCHS):
      history = model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)
      model.reset_states()
    for k in range(N_DATES_TO_PREDICT):
      closing_price = model.predict(X_test)
      #update X_test
      X_test = np.delete(X_test, [[0]])
      X_test = np.append(X_test, closing_price[0][0])
      X_test = np.reshape(X_test, (1,RANGE,1))

      closing_price = SCALER.inverse_transform(closing_price)
      prediction.append(closing_price[0])
    print(np.array(prediction))
  '''
      #test_mse.append(mean_squared_error(y_test, closing_price))
      #train_mse.append(history.history['loss'][0])

    #plt.plot(test_mse, color='blue')
    #plt.plot(train_mse, color='orange')
  #plt.show()

  '''


  index_to_split = int(TRAIN_DATA_PERCENTAGE * len(new_data.values))
  train = new_data[:index_to_split]
  valid = new_data[index_to_split:index_to_split+N_DATES_TO_PREDICT]

  print("The first test date: ", new_data.index[index_to_split])

  #closing_price = SCALER.inverse_transform(closing_price)

  #MSE
  #mse = mean_squared_error(valid[['close']], closing_price)
  #print("Mean Squared Error = " + str(mse))


  #for plotting
  print (valid[['close']])
  valid['Predictions'] = np.array(prediction)

  plt.plot(train['close'])
  plt.plot(valid[['close','Predictions']])
  plt.show()




if __name__ == "__main__":
  main()
