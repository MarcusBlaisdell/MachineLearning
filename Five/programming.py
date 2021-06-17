import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

# create global arrays to store data and labels:
dataList = []
labelList = []


############################################
# runMLP function
# Multi-Layer Perceptron
# Use relu activation function
# use alpha = 1e-5

def runMLP (theData, theLabel):
  clf = MLPClassifier (activation='relu', solver='lbfgs', alpha=1e-5,
                      hidden_layer_sizes=(5,2), random_state=1)

  clf.fit (theData, theLabel)

  return clf


# end runMLP function
############################################


############################################
# runLR function
# Logistic Regression classifer
# maximum iterations = 50

def runLR (theData, theLabel):
  clf = LogisticRegression (random_state=0, solver='lbfgs',
                           multi_class='multinomial', max_iter=50)

  clf.fit (theData, theLabel)

  return clf


# end runLR function
############################################


############################################
# runRF function
# Random Forest classifier
# 10 decision trees (n_estimators = 10)


def runRF (theData, theLabel):
  clf = RandomForestClassifier(n_estimators=10, max_depth=2,
                             random_state=0)

  clf.fit (theData, theLabel)

  return clf


# end runRF function
############################################


############################################
# getLabel function
# accepts an array of 10 characters
# and determines which position the '1' is in
# and returns that array position as the
# digit label:

def getLabel (myLabel):

    # initialize with impossible value to detect errors:

  theLabel = 12

    # iterate through the list looking for the '1':
    # return as soon as we find it (short circuit evaluation)

  for label in range (10):

    if myLabel[label] == '1':

        # if we found the '1'
        # return our current position
        # as the digit label:

      return label

      # else, return our bogus value 12:

  return theLabel

# end getLabel function
############################################



############################################
# getData function
# read data from file
# split into data and label portions
# and input into appropriate
# global arrays:

def getData ():

    # open the file for read:

  inFile = open('semeion.data', 'r')
  #inFile = open('/content/gdrive/My Drive/437/HW5/Samples.data', 'r')

  myString = inFile.readline()

  myList = myString.split (" ")

  myData = myList[:-11]

  while (myString):

    myList = myString.split (" ")

    myData = myList[:-11]
    dataList.append (myData)

    '''
    # Print the data, 16 rows of 16 elements per row:
    for i in range (16):
      print ("")
      for j in range (16):
        print (myData[(i * 16) + j], "",  end="")

    print ("")
    '''

      # Convert the labels to a single number, 0-9
      # that represents the digit that the sample
      # has been labelled as

    myLabel = myList[-11:]
    theLabel = getLabel(myLabel)
    labelList.append (theLabel)

    #print ("theLabel: ", theLabel)

    myString = inFile.readline ()


# end getData function
############################################


############################################
# formatData function
# Data is read in as strings,
# Need to convert to float:

def formatData ():
  lineNum = 0

  for line in dataList:
    for element in range (len(line)):
      dataList[lineNum][element] = float(dataList[lineNum][element])
    lineNum += 1


# end formatData function
############################################


#########################################################
# myKFolds function:
# divides the list into K, mostly equal parts
# and returns a list with all of the indexes

def myKFolds (theSize, theK):
  theList = []
  spacing = int (theSize / theK)

  for i in range (theK - 1):
    theList.append([i * spacing, i * spacing + spacing - 1])

  theList.append([(theK - 1) * spacing, theSize])

  return theList

#
#########################################################


#########################################################
# vote function
# accepts 3 predictions,
# returns the prediction that occurs the most frequently
# if no predictions are in common, return the prediction
# of the logistic regression classifier (pred2) because
# experimentation shows that it produces the highest
# accuracy (~91% vs. ~68% if we choose either of the
# other two classifiers)

def vote (pred1, pred2, pred3):

  # if they all match, trivial

  if (pred1 == pred2 == pred3):
    return pred1

  # otherwise, do an actual evaluation:
  # initialize count1 to 1 because it is the count
  # of the first prediction so since there is a first
  # prediction (pred1), we already know there is at
  # least 1 pred1
  # initialize count2 and count3 to zero, only increment them if
  # we encounter predictions that don't match pred1

  count1 = 1
  count2 = 0
  count3 = 0

  if pred2 == pred1:
    count1 += 1
  else:
    count2 += 1

  if pred3 == pred1:
    count1 += 1
  elif pred3 == pred2:
    count2 += 1
  else:
    count3 += 1

  # if any count is greater than or equal to 2,
  # it has the majority of 3 so return that prediction
  # but if neither count1 or count2 are 2 or more,
  # then there are no common predictions so
  # favor logistic regression as the main predictor and return
  # pred2

  if count1 > 1:
    return pred1
  elif count2 > 1:
    return pred2
  else:
    return pred2

# end vote function
#########################################################



############################################
# main function

def main ():

    # Get the data:

  getData ()

    # format the data:

  formatData ()

  #print ("data size: ", len(dataList))
  #print ("label size: ", len(labelList))

  print ("Run with full resolution of 256:\n\n")
      # use 3-folds:

  numFolds = 3

  myList = myKFolds (len(dataList), numFolds)

    # Traverse the folds:

  for i in range (len(myList)):

    testData = dataList[myList[i][0]:myList[i][1] + 1]
    testLabel = labelList[myList[i][0]:myList[i][1] + 1]

      # initialize trainset with i + 1 % len(list)
      # to get the next index in a circular array:

    trainData = dataList[myList[(i + 1) % len(myList)][0]:myList[(i + 1) % len(myList)][1] + 1]
    trainLabel = labelList[myList[(i + 1) % len(myList)][0]:myList[(i + 1) % len(myList)][1] + 1]

      # now, concatenate remaining indexes that are not the test index

    for j in range (1, len(myList) - 1):

      trainData = np.concatenate ((trainData, dataList[myList[(i + 1 + j) % len(myList)][0]:myList[(i + 1 + j) % len(myList)][1] + 1]), axis=0)
      trainLabel = np.concatenate ((trainLabel, labelList[myList[(i + 1 + j) % len(myList)][0]:myList[(i + 1 + j) % len(myList)][1] + 1]), axis=0)

      # train the data using each of the classifiers:

    mlpCLF = runMLP (dataList, labelList)
    lrCLF = runLR (dataList, labelList)
    rfCLF = runRF (dataList, labelList)

      # test the model using voting:

    correct = 0

    for testIndex in range (len(testData)):

        # get the prediction of each classifier:

        pred1 = mlpCLF.predict ([dataList[testIndex]])
        pred2 = lrCLF.predict ([dataList[testIndex]])
        pred3 = rfCLF.predict ([dataList[testIndex]])

        #print ("mlp: ", pred1, " lr: ", pred2, " rf: ", pred3)

        # get the prediction that has the most votes:

        thePred = vote (pred1, pred2, pred3)

        # evaluate the prediction:

        if thePred == labelList[testIndex]:
          correct += 1

    theAccuracy = 100 * (float (correct) / len(testData) )

    print ("Accuracy: ", theAccuracy )

  ### Repeat for PCA:

  print ("\n\nrepeat with dimensionality reduction to 96:\n\n")

  ### PCA:

  ### By experimentation, PCA Value : Accuracy
  # 128 : 95.85687382297552
  # 96 : 98.68173258003766
  # 64 : 97.92843691148776
  # 32 : 94.35028248587571
  # 16 : 88.51224105461394

  pca = PCA(96)

  print ("dataList[0]: ")
  print (dataList[0])
  newDataList = pca.fit_transform(dataList)

  print ("newDataList[0]: ")
  print (newDataList[0])

      # use 3-folds:

  numFolds = 3

  myList = myKFolds (len(newDataList), numFolds)

    # Traverse the folds:

  for i in range (len(myList)):

    testData = newDataList[myList[i][0]:myList[i][1] + 1]
    testLabel = labelList[myList[i][0]:myList[i][1] + 1]

      # initialize trainset with i + 1 % len(list)
      # to get the next index in a circular array:

    trainData = newDataList[myList[(i + 1) % len(myList)][0]:myList[(i + 1) % len(myList)][1] + 1]
    trainLabel = labelList[myList[(i + 1) % len(myList)][0]:myList[(i + 1) % len(myList)][1] + 1]

      # now, concatenate remaining indexes that are not the test index

    for j in range (1, len(myList) - 1):

      trainData = np.concatenate ((trainData, newDataList[myList[(i + 1 + j) % len(myList)][0]:myList[(i + 1 + j) % len(myList)][1] + 1]), axis=0)
      trainLabel = np.concatenate ((trainLabel, labelList[myList[(i + 1 + j) % len(myList)][0]:myList[(i + 1 + j) % len(myList)][1] + 1]), axis=0)

      # train the data using each of the classifiers:

    mlpCLF = runMLP (newDataList, labelList)
    lrCLF = runLR (newDataList, labelList)
    rfCLF = runRF (newDataList, labelList)

      # test the model using voting:

    correct = 0

    for testIndex in range (len(testData)):

        # get the prediction of each classifier:

        pred1 = mlpCLF.predict ([newDataList[testIndex]])
        pred2 = lrCLF.predict ([newDataList[testIndex]])
        pred3 = rfCLF.predict ([newDataList[testIndex]])

        #print ("mlp: ", pred1, " lr: ", pred2, " rf: ", pred3)

        # get the prediction that has the most votes:

        thePred = vote (pred1, pred2, pred3)

        # evaluate the prediction:

        if thePred == labelList[testIndex]:
          correct += 1

    theAccuracy = 100 * (float (correct) / len(testData) )

    print ("Accuracy: ", theAccuracy )

# end main function
############################################




if __name__ == "__main__":
  main ()
