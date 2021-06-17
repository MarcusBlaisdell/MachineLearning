#########################################################
#

def myKFolds (theSize, theK):
  theList = []
  spacing = theSize / theK

  for i in range (theK - 1):
    theList.append([i*spacing, i*spacing + spacing - 1])

  theList.append([(theK - 1) * spacing, theSize - 1])

  return theList

#
#########################################################

#########################################################
# Main function:

if __name__ == "__main__":

    myList = myKFolds (51337, 10)

    print myList

    print ("\n\n")

    test = [(0,0)]
    train = []

    for i in range(len(list)):
        test[0] =


# end Main function
