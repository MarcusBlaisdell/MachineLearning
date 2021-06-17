from __future__ import absolute_import, division, print_function

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.decomposition import PCA


########################################
# build_model function
# build the model:

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model

# end build_model function
########################################

########################################
# plot_history function

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [close]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.ylim([0,1])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$close^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.ylim([0,1])
  plt.legend()
  plt.show()

# end plot_history function
########################################

# Check the tensorflow version, if desired:

#print(tf.__version__)



column_names = ['date', 'close', 'volume', 'open', 'high', 'low']

raw_dataset = pd.read_csv ("normalized_Microsoft_asc.csv",
                            names=column_names, na_values = "?",
                            comment='\t', sep=",", skipinitialspace=True)


dataset = raw_dataset.copy()

print ("********************")
'''
print ("dataset:")
print (dataset)

printDataset = list(raw_dataset)
print ("printDataset:")
print (printDataset)

pca = PCA(2)

newData = pca.fit_transform(dataset)

newDataList = list(newData)

print ("newDataList:")
print (newDataList)
'''
print ("********************")


#print (dataset.tail() )

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

#sns.pairplot(train_dataset[["date", "close", "volume", "open", "high", "low"]], diag_kind="kde")

train_stats = train_dataset.describe()
train_stats.pop("close")
train_stats = train_stats.transpose()
print (train_stats)

train_labels = train_dataset.pop('close')
test_labels = test_dataset.pop('close')

'''
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
'''

model = build_model()

model.summary()

example_batch = train_dataset[:10]
example_result = model.predict(example_batch)
print (example_result)

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 100

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)


history = model.fit(
  train_dataset, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])

print ("\n")

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

print (hist.tail())

plot_history(history)

#print (test_dataset)
loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=0)

print("Testing set Mean Abs Error: {:5.2f} close".format(mae))

test_predictions = model.predict(test_dataset).flatten()

tempPred = list(test_predictions)
tempLabel = list(test_labels)

for i in range(len(tempPred)):
    print ("label: ", tempLabel[i], " - prediction: ", tempPred[i])
    #print (" - prediction: ", temp[i])

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [close]')
plt.ylabel('Predictions [close]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()

error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [close]")
_ = plt.ylabel("Count")
plt.show()
