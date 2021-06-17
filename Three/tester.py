import numpy as np
from sklearn.linear_model import Perceptron
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from scipy import stats

### Create a global array to hold all of the data files:

fileList = ["monks-1.csv", "monks-2.csv", "monks-3.csv"]

tVals1 = []
tVals2 = []

#########################################################
# formatData function:
# the monks data is arranged with the labels in the
# zero position and the features in the remaining positions
# This function reads these files
# and returns the attributes and labels
# as separate arrays:

def formatData (Data):
  X = []
  y = []

  for i in range (len(Data)):
    y.append (Data[i][0])
    X.append (Data[i][1:len(Data[i])])

  return X, y

# end formatData function
#########################################################



#########################################################
### Perceptron

def runPerceptron ():
  #########################################################

  accuracyCount = 0

    ### Print perceptron header:

  print ("\n\tperceptron\n")

    # run perceptron, max iterations = 50
    # using k-fold evaluation:

    ### Create the perceptron:

  clf = Perceptron(tol=1e-3, random_state=0, max_iter=50)

  ### Load the data

  Data = np.loadtxt(fname=fileList[0], delimiter=',')
  Data = np.concatenate ((Data, np.loadtxt(fname=fileList[1], delimiter=',')), axis=0)
  Data = np.concatenate ((Data, np.loadtxt(fname=fileList[2], delimiter=',')), axis=0)

    ### use k-fold evaluation:

    ### pass 1:
  #print ("\nPass 1:\n")

  testData = Data[0:570]
  trainData = Data[570:1140]

      ### Format the Training data:

  X, y = formatData (trainData)

    ### learn the training data:

  clf.fit(X,y)

    ### Format the test data:

  X, y = formatData (testData)

    ### print the results of the test:

  #print ("Accuracy: ", 100 * (clf.score (X,y) ), "%\n" )
  accuracyCount += ( 100 * (clf.score (X,y) ) )

  ### pass 2:
  #print ("\nPass 2:\n")

  testData = Data[570:1140]
  trainData = np.concatenate ((Data[0:570], Data[1140:1711]), axis=0 )

      ### Format the Training data:

  X, y = formatData (trainData)

    ### learn the training data:

  clf.fit(X,y)

    ### Format the test data:

  X, y = formatData (testData)

    ### print the results of the test:

  #print ("Accuracy: ", 100 * (clf.score (X,y) ), "%\n" )
  accuracyCount += ( 100 * (clf.score (X,y) ) )

  ### pass 3:
  #print ("\nPass 3:\n")

  testData = Data[1140:1711]
  trainData = Data[0:1140]

      ### Format the Training data:

  X, y = formatData (trainData)

    ### learn the training data:

  clf.fit(X,y)

    ### Format the test data:

  X, y = formatData (testData)

    ### print the results of the test:

  #print ("Accuracy: ", 100 * (clf.score (X,y) ), "%\n" )
  accuracyCount += ( 100 * (clf.score (X,y) ) )


  print ("Accuracy of 3-fold = ", float(accuracyCount) / float(3) )

  ### end k-fold loop
  #########################################################


  #########################################################
  ### begin leave-one-out

  accuracyCount = 0.0

  for i in range(len(Data)):
    trainData = np.delete (Data, i, 0)
    testData = Data[i:i+1]

    X, y = formatData(trainData)

    clf.fit (X, y)

    X, y = formatData(testData)

    #print ("Accuracy: ", 100 * (clf.score (X,y) ), "%\n" )
    accuracyCount += 100 * (clf.score (X,y) )

    tVals1.append (100 * (clf.score (X,y)))

  print ("Accuracy of LOO: ", accuracyCount / float(len(Data)))

  ### end leave-one-out
  #########################################################



  ### end runPerceptron function
  #########################################################



#########################################################
### Decision Trees:

def runDecisionTrees ():
  #########################################################
  ### Print decision tree header:

  print ("\n\tDecision Trees\n")


    ### create a decision tree classifier:

  clf = tree.DecisionTreeClassifier ()

    ### use 3-fold evaluation:

  #for i in range (3):

    ### load first training data set:

  Data = np.loadtxt(fname=fileList[0], delimiter=',')

    ### Decision Trees does not have a partial fit option so we need to
    ### load both training datasets at once:

  Data = np.concatenate ((Data, np.loadtxt(fname=fileList[1], delimiter=',')), axis=0)
  Data = np.concatenate ((Data, np.loadtxt(fname=fileList[2], delimiter=',')), axis=0)

  '''
    ### Format the data:

  X, y = formatData (Data)

    ### train:

  clf.fit(X,y)

    ### load the test data set:

  Data = np.loadtxt(fname='/content/gdrive/My Drive/437/HW3/' + fileList[i], delimiter=',')



    ### Format the data:

  X, y = formatData (Data)

    ### print the results of the test:

  print ("train set 1: ", fileList[((i+1)%3)])
  print ("train set 2: ", fileList[((i+2)%3)])
  print ("testset : ", fileList[i])
  print ("Accuracy: ", 100 * (clf.score (X,y) ), "%\n" )
  '''

  accuracyCount = 0.0

    ### pass 1:
  #print ("\nPass 1:\n")

  testData = Data[0:570]
  trainData = Data[570:1140]

      ### Format the Training data:

  X, y = formatData (trainData)

    ### learn the training data:

  clf.fit(X,y)

    ### Format the test data:

  X, y = formatData (testData)

    ### print the results of the test:

  #print ("Accuracy: ", 100 * (clf.score (X,y) ), "%\n" )
  accuracyCount += ( 100 * (clf.score (X,y) ) )

  ### pass 2:
  #print ("\nPass 2:\n")

  testData = Data[570:1140]
  trainData = np.concatenate ((Data[0:570], Data[1140:1711]), axis=0 )

      ### Format the Training data:

  X, y = formatData (trainData)

    ### learn the training data:

  clf.fit(X,y)

    ### Format the test data:

  X, y = formatData (testData)

    ### print the results of the test:

  #print ("Accuracy: ", 100 * (clf.score (X,y) ), "%\n" )
  accuracyCount += ( 100 * (clf.score (X,y) ) )

  ### pass 3:
  #print ("\nPass 3:\n")

  testData = Data[1140:1711]
  trainData = Data[0:1140]

      ### Format the Training data:

  X, y = formatData (trainData)

    ### learn the training data:

  clf.fit(X,y)

    ### Format the test data:

  X, y = formatData (testData)

    ### print the results of the test:

  #print ("Accuracy: ", 100 * (clf.score (X,y) ), "%\n" )
  accuracyCount += ( 100 * (clf.score (X,y) ) )


  print ("Accuracy of 3-fold = ", float(accuracyCount) / float(3) )

  ### end 3-fold loop
  #########################################################


  #########################################################
  ### begin leave-one-out

  accuracyCount = 0.0

  for i in range(len(Data)):
    trainData = np.delete (Data, i, 0)
    testData = Data[i:i+1]

    X, y = formatData(trainData)

    clf.fit (X, y)

    X, y = formatData(testData)

    #print ("Accuracy: ", 100 * (clf.score (X,y) ), "%\n" )
    accuracyCount += 100 * (clf.score (X,y) )

  print ("Accuracy of LOO: ", accuracyCount / float(len(Data)))

  ### end leave-one-out
  #########################################################



  ### end runDecisionTrees function
  #########################################################



#########################################################
### k-nearest neighbor:

def runKNN ():
  #########################################################
  ### Print knn header:

  print ("\n\tk-nearest neighbor\n")

    ### Create a nearest-neighbor classifier

  clf = KNeighborsClassifier (n_neighbors=3)

  ### Use 3-fold:

  #for i in range (3):

    ### Load first data set:

  Data = np.loadtxt(fname=fileList[0], delimiter=',')

    ### Append the second data set:

  Data = np.concatenate ((Data, np.loadtxt(fname=fileList[1], delimiter=',')), axis=0)
  Data = np.concatenate ((Data, np.loadtxt(fname=fileList[2], delimiter=',')), axis=0)

  '''
    ### Format the data:

  X, y = formatData (Data)

    ### Train:

  clf.fit (X, y)

    ### Load test data set:

  Data = np.loadtxt(fname='/content/gdrive/My Drive/437/HW3/' + fileList[i], delimiter=',')

  ### Format the data:

  X, y = formatData (Data)

    ### calculate prediction accuracy:

  correct = 0

  for j in range (len(Data)):
    if (clf.predict ( [X[j]]) ) == y[j]:
      correct += 1

    ### print the results of the test:

  print ("train set 1: ", fileList[((i+1)%3)])
  print ("train set 2: ", fileList[((i+2)%3)])
  print ("testset : ", fileList[i])

  print ("Accuracy: ", (100 * (correct / float(len(Data) ) ) ), "%\n")
  '''

  accuracyCount = 0.0

    ### pass 1:
  #print ("\nPass 1:\n")

  testData = Data[0:570]
  trainData = Data[570:1140]

      ### Format the Training data:

  X, y = formatData (trainData)

    ### learn the training data:

  clf.fit(X,y)

    ### Format the test data:

  X, y = formatData (testData)


      ### calculate prediction accuracy:

  correct = 0

  for j in range (len(testData)):
    if (clf.predict ( [X[j]]) ) == y[j]:
      correct += 1

  accuracyCount += (100 * (correct / float(len(testData) ) ) )

  ### pass 2:
  #print ("\nPass 2:\n")

  testData = Data[570:1140]
  trainData = np.concatenate ((Data[0:570], Data[1140:1711]), axis=0 )

      ### Format the Training data:

  X, y = formatData (trainData)

    ### learn the training data:

  clf.fit(X,y)

    ### Format the test data:

  X, y = formatData (testData)

    ### calculate prediction accuracy:

  correct = 0

  for j in range (len(testData)):
    if (clf.predict ( [X[j]]) ) == y[j]:
      correct += 1

  accuracyCount += (100 * (correct / float(len(testData) ) ) )

  ### pass 3:
  #print ("\nPass 3:\n")

  testData = Data[1140:1711]
  trainData = Data[0:1140]

      ### Format the Training data:

  X, y = formatData (trainData)

    ### learn the training data:

  clf.fit(X,y)

    ### Format the test data:

  X, y = formatData (testData)

    ### calculate prediction accuracy:

  correct = 0

  for j in range (len(testData)):
    if (clf.predict ( [X[j]]) ) == y[j]:
      correct += 1

  accuracyCount += (100 * (correct / float(len(testData) ) ) )


  print ("Accuracy of 3-fold = ", float(accuracyCount) / float(3) )


  ### end 3-fold loop
  #########################################################

  #########################################################
  ### begin leave-one-out

  accuracyCount = 0.0

  for i in range(len(Data)):
    trainData = np.delete (Data, i, 0)
    testData = Data[i:i+1]

    X, y = formatData(trainData)

    clf.fit (X, y)

    X, y = formatData(testData)

    #print ("Accuracy: ", 100 * (clf.score (X,y) ), "%\n" )
    if (clf.predict ( [X[0]] ) ) == y[0]:
      correct += 1

    #accuracyCount += (100 * (correct / float(len(testData) ) ) )

  print ("Accuracy of LOO: ", ( 100 * (correct / float(len(Data) ) )))

  ### end leave-one-out
  #########################################################

  ### end runKNN function
  #########################################################



def runNaiveBayes ():
  #########################################################
  ### Print naive bayes header:

  print ("\n\tnaive bayes\n")

    ### Create a naive bayes classifier (using Gaussian):

  clf = GaussianNB ()

  ### Use 3-fold:

  for i in range (3):

      ### Load first data set:

    Data = np.loadtxt(fname=fileList[0], delimiter=',')

      ### Append the second data set:

    Data = np.concatenate ((Data, np.loadtxt(fname=fileList[1], delimiter=',')), axis=0)
    Data = np.concatenate ((Data, np.loadtxt(fname=fileList[2], delimiter=',')), axis=0)

    '''
      ### Format the data:

    X, y = formatData (Data)

      ### Train:

    clf.fit (X, y)

      ### Load test data set:

    Data = np.loadtxt(fname='/content/gdrive/My Drive/437/HW3/' + fileList[i], delimiter=',')

    ### Format the data:

    X, y = formatData (Data)

    '''

    accuracyCount = 0.0

    ### pass 1:
  #print ("\nPass 1:\n")

  testData = Data[0:570]
  trainData = Data[570:1140]

      ### Format the Training data:

  X, y = formatData (trainData)

    ### learn the training data:

  clf.fit(X,y)

    ### Format the test data:

  X, y = formatData (testData)


      ### calculate prediction accuracy:

  correct = 0

  for j in range (len(testData)):
    if (clf.predict ( [X[j]]) ) == y[j]:
      correct += 1

  accuracyCount += (100 * (correct / float(len(testData) ) ) )

  ### pass 2:
  #print ("\nPass 2:\n")

  testData = Data[570:1140]
  trainData = np.concatenate ((Data[0:570], Data[1140:1711]), axis=0 )

      ### Format the Training data:

  X, y = formatData (trainData)

    ### learn the training data:

  clf.fit(X,y)

    ### Format the test data:

  X, y = formatData (testData)

    ### calculate prediction accuracy:

  correct = 0

  for j in range (len(testData)):
    if (clf.predict ( [X[j]]) ) == y[j]:
      correct += 1

  accuracyCount += (100 * (correct / float(len(testData) ) ) )

  ### pass 3:
  #print ("\nPass 3:\n")

  testData = Data[1140:1711]
  trainData = Data[0:1140]

      ### Format the Training data:

  X, y = formatData (trainData)

    ### learn the training data:

  clf.fit(X,y)

    ### Format the test data:

  X, y = formatData (testData)

    ### calculate prediction accuracy:

  correct = 0

  for j in range (len(testData)):
    if (clf.predict ( [X[j]]) ) == y[j]:
      correct += 1

  accuracyCount += (100 * (correct / float(len(testData) ) ) )


  print ("Accuracy of 3-fold = ", float(accuracyCount) / float(3) )


  '''
      ### calculate prediction accuracy:

    correct = 0

    for j in range (len(Data)):
      if (clf.predict ( [X[j]]) ) == y[j]:
        correct += 1

      ### print the results of the test:

    print ("train set 1: ", fileList[((i+1)%3)])
    print ("train set 2: ", fileList[((i+2)%3)])
    print ("testset : ", fileList[i])

    print ("Accuracy: ", (100 * (correct / float(len(Data) ) ) ), "%\n")
  '''

  ### end 3-fold loop
  #########################################################


  #########################################################
  ### begin leave-one-out

  accuracyCount = 0.0

  for i in range(len(Data)):
    trainData = np.delete (Data, i, 0)
    testData = Data[i:i+1]

    X, y = formatData(trainData)

    clf.fit (X, y)

    X, y = formatData(testData)

    #print ("Accuracy: ", 100 * (clf.score (X,y) ), "%\n" )
    if (clf.predict ( [X[0]] ) ) == y[0]:
      correct += 1

    tVals2.append (100 * (correct / float(len(Data))))

    #accuracyCount += (100 * (correct / float(len(testData) ) ) )

  print ("Accuracy of LOO: ", ( 100 * (correct / float(len(Data) ) )))

  ### end leave-one-out
  #########################################################


  ### end runNaiveBayes function
  #########################################################


#########################################################
### t-test:

def runTTest ():
  #########################################################
  ### Print t-test header:

  print ("\n\tt-test\n")

  np.random.seed(1)

  t, p = stats.ttest_rel (tVals1, tVals2)

  print ("t: ", t, " - p: ", p)

  ### end runTTeset function
  #########################################################


#########################################################
# Main function:

if __name__ == "__main__":

    ### run the perceptron:

  runPerceptron ()

    ### run the decision trees:

  runDecisionTrees ()


    ### run the nearest neighbors:

  runKNN ()

    ### run the naive bayes:

  runNaiveBayes ()

    ### run the t-test performance measure:

  runTTest ()
  print ("len(tVals1): ")
  print (len(tVals1) )
  print ("len(tVals2): ")
  print (len(tVals1) )


# end Main function
#########################################################
