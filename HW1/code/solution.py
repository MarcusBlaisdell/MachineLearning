import numpy as np 
from helper import *

'''
Homework1: perceptron classifier
'''
def sign(x):
	return 1 if x > 0 else -1

#-------------- Implement your code Below -------------#
# Data for plotting
t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(2 * np.pi * t)

def show_images(data):
    '''
    This function is used for plot image and save it.

    Args:
        data: Two images from train data with shape (2, 16, 16). The shape represents total 2
        images and each image has size 16 by 16. 

    Returns:
        Do not return any arguments, just save the images you plot for your report.
    '''
    
    ### familiarize myself with the data by printing it out:
    
    #print (data)
    
    ### create a figure to plot the greyscale values:
    
    fig, ax = plt.subplots()
    
    ### imshow interprets the matrix for me:
    
    ax.imshow (data[0])
    
    ### save the image to file:
    
    fig.savefig("data_0.png")
    
    ### repeat for the second image:
    
    ax.imshow (data[1])
    
    fig.savefig("data_1.png")
    


def show_features(data, label):
    '''
    This function is used for plot a 2-D scatter plot of the features 
    and save it. 

    Args:
        data: train features with shape (1561, 2). The shape represents 
            total 1561 samples and 
            each sample has 2 features.
            label: train data's label with shape (1561,1). 
                1 for digit number 1 and -1 for digit number 5.

    Returns:
        Do not return any arguments, just save the 2-D scatter plot of the features you plot for your report.
        '''
    
    
    ### Familiarize myself with the data by printing it out:
    
    #print ("data")
    #   print (data)
    #print ("label")
    #print (label)
    
    ### create a figure to plot the scatter plot in
    
    fig, ax = plt.subplots()
    
    ### use a variable to print the appropriate data point:
    
    i = 0
   
    ### iterate through each data point, if it is labeled as 1, 
    ### print a red *, otherwise, print a blue +
    
    for x in data:
        if (label[i] == 1):
            ax.scatter(data[i][0], data[i][1], marker='*', c='red')
        else:
            ax.scatter(data[i][0], data[i][1], marker='+', c='blue')
        i = i + 1
    
    ### save the image to file:
    
    fig.savefig("scatter.png")


def perceptron(data, label, max_iter, learning_rate):
    '''
    The perceptron classifier function.

    Args:
    data: train data with shape (1561, 3), which means 1561 samples and 
        each sample has 3 features.(1, symmetry, average intensity)
    label: train data's label with shape (1561,1). 
        1 for digit number 1 and -1 for digit number 5.
    max_iter: max iteration numbers
    learning_rate: learning rate for weight update
	
    Returns:
        w: the seperator with shape (1, 3). 
        You must initialize it with w = np.zeros((1,d))
    '''
    
    ### Print the inputs to familiarize myself with the data:
    
    '''
    print ("perceptron function")
    print (data)
    print ("label:")
    print (label)
    print ("max_iter:")
    print (max_iter)
    print ("learning_rate")
    print (learning_rate)
    '''
    
    ### perceptron algorithm:
    '''
    1. initialize w to zero, w is a vector
    2. select any mis-classified data point, (x_n, y_n), let it be
        (x_*,y_*)
    3. update w by w(t + 1) = w(t) + x_*y_*
    4. run until there are no mis-classified examples or
        count = max_iter
    5. return w
    '''
    
    x = 0
    
    ### initialize to zeroes, w, as an array of size three:
    
    w = np.zeros((1,3))
    
    # modify w while we have mis-classified data points:
    
    for i in range(max_iter):
        # calculate h(x) as y
        
        y = sign(w[0][0] * data[x][0] + w[0][1] * data[x][1] + w[0][2] * data[x][2])
    
        # find a mis-classified data point
        # move past any properly classified points:
        
        while y == label[x]:
            x = x + 1
            if x > len(data) - 1:
                # if we go past the end of the array, start over:
                x = 0
            y = sign(w[0][0] * data[x][0] + w[0][1] * data[x][1] + w[0][2] * data[x][2])
            
        # update w
        w[0][0] = w[0][0] + data[x][0] * label[x]
        w[0][1] = w[0][1] + data[x][1] * label[x]
        w[0][2] = w[0][2] + data[x][2] * label[x]
        
    return w
    

def show_result(data, label, w):
    '''
    This function is used for plot the test data with the separators and 
    save it.
	
    Args:
    data: test features with shape (424, 2). The shape represents total 
        424 samples and 
        each sample has 2 features.
    label: test data's label with shape (424,1). 
        1 for digit number 1 and -1 for digit number 5.
	
    Returns:
    Do not return any arguments, just save the image you plot for your report.
    '''
    #print ("show result, data")
    #print (data)
    
    ### create a figure to plot the scatter plot in
    
    fig, ax = plt.subplots()
    
    ### use a variable to print the appropriate data point:
    
    i = 0
   
    ### iterate through each data point, if it is labeled as 1, 
    ### print a red *, otherwise, print a blue +
    
    for x in data:
        if (label[i] == 1):
            ax.scatter(data[i][0], data[i][1], marker='*', c='red')
        else:
            ax.scatter(data[i][0], data[i][1], marker='+', c='blue')
        i = i + 1
    
    print ("w: ", w)
    ax.plot([0,w[0][1]],[0,w[0][2]])
    
    ### save the image to file:
    
    fig.savefig("scatter_test.png")    
    


#-------------- Implement your code above ------------#
def accuracy_perceptron(data, label, w):
	n, _ = data.shape
	mistakes = 0
	for i in range(n):
		if sign(np.dot(data[i,:],np.transpose(w))) != label[i]:
			mistakes += 1
	return (n-mistakes)/n


def test_perceptron(max_iter, learning_rate):
	#get data
	traindataloc,testdataloc = "../data/train.txt", "../data/test.txt"
	train_data,train_label = load_features(traindataloc)
	test_data, test_label = load_features(testdataloc)
	#train perceptron
	w = perceptron(train_data, train_label, max_iter, learning_rate)
	train_acc = accuracy_perceptron(train_data, train_label, w)	
	#test perceptron model
	test_acc = accuracy_perceptron(test_data, test_label, w)
	return w, train_acc, test_acc


