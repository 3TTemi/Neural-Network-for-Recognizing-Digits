#Imports needed for math symbols and randomness
import random 
import math

class neuralNetwrok(): 

    #initializing all needed variabes for neural network
    def __init__(self,inputSize,hiddenSize,outputSize, learnRate, numIterate, inputSet, resultSet):
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.learnRate = learnRate 
        self.numIterate = numIterate
        self.inputSet = inputSet
        self.resultSet = resultSet
        self.toHiddenWeights = []
        self.toOutputWeights = []

    #Weight 2d arrays are broken down so layer 2 is rows and layer 1 is column
    def initWeights(self):
        #iterating to set number of blank arrays for hidden nodes
        for i in range(self.hiddenSize):
            self.toHiddenWeights.append([])
        #intializing all weihts in input-hidden 2d array to small random value
        for i in range(self.hiddenSize):
            for j in range(self.inputSize):
                self.toHiddenWeights[i].append(round(random.uniform(-0.2,0.2), 2))
        #iterating to set number of blank arrays for output nodes
        for i in range(self.outputSize):
            self.toOutputWeights.append([])
        #intializing all weihts in hidden-output 2d array to small random value
        for i in range(self.outputSize):
            for j in range(self.hiddenSize):
                self.toOutputWeights[i].append(round(random.uniform(-0.2,0.2), 2))

    #creating all node objects given their sizes
    def createNodes(self): 
        self.inputList = []
        self.hiddenList = []
        self.outputList = []
        for i in range(self.inputSize): 
            self.inputList.append(inputNode(None)) 
        for i in range(self.hiddenSize): 
            self.hiddenList.append(hiddenNode())
        for i in range(self.outputSize): 
            self.outputList.append(outputNode()) 
            
    #Intializing all of the object values for input,hidden, and output nodes given single input and output 
    def initFile(self, inputGrid, expectedNum): 
        #used to find the which number it expected for the input (Format for validation and training files)
        index = expectedNum.find('1')
        for i in range(self.inputSize):
            self.inputList[i].value = int(inputGrid[i])
        for i in range(self.hiddenSize):
            self.hiddenList[i].output = 0
        for i in range(self.outputSize):
            self.outputList[i].expect = 0
            self.outputList[i].output = 0
        # set the expected value of the correct numbers outputnode to 1 
        self.outputList[index].expect = 1

    #Foward propogation algorithm to sovle for all inputs 
    def forwardProp(self): 
        #Iterating through all hidden nodes nodes and summing the product of weights and input nodes 
        for i in range(len(self.toHiddenWeights)):
            for j in range(len(self.inputList)):
                if self.hiddenList[i].output == None:
                    self.hiddenList[i].output = 0
                self.hiddenList[i].output += self.toHiddenWeights[i][j] * self.inputList[j].value
            #Rounded Sigmoid function to get final reuslt
            self.hiddenList[i].output = round((1 / (1 + (math.e ** -self.hiddenList[i].output))),2)

        #Iterating through all output nodes and summing the product of weights and hidden nodes
        for i in range(len(self.toOutputWeights)): 
            for j in range(len(self.hiddenList)): 
                if self.outputList[i].output == None:
                    self.outputList[i].output = 0
                self.outputList[i].output += self.toOutputWeights[i][j] * self.hiddenList[j].output
            #Rounded Sigmoid function to get final reuslt
 
            self.outputList[i].output = round((1 / (1 + (math.e ** -self.outputList[i].output))),2)
    #Back Proprogation to change output weights and get closer to correctly predicitng algorithm
    def backProp(self): 
        #Iterating backwards from output to hidden weights 
        for row in range(len(self.toOutputWeights)): 
            for col in range(len(self.toOutputWeights[0])):  
                #Calculating error at output Node
                self.outputList[row].error = (self.outputList[row].output) * (1 - (self.outputList[row].output)) * (self.outputList[row].expect - self.outputList[row].output)
                nodeInput = self.hiddenList[col].output 
                #Calculating weight change necessary for output-hidden weight and changing accordingly
                weightChangeO = self.learnRate * self.outputList[row].error * nodeInput  
                self.toOutputWeights[row][col] += weightChangeO

        #Iterating backwards from hidden to input weights 
        for row in range(len(self.toHiddenWeights)): 
            for col in range(len(self.toHiddenWeights[0])): 
                outputSum = 0
                #Iterating through outputNodes to find sum of the product of weights adn errors
                for outIndex in range(len(self.outputList)):
                    outputSum += self.toOutputWeights[outIndex][row] * self.outputList[outIndex].error3
                nodeInput = self.inputList[col].value
                hiddenOutput = self.hiddenList[row].output 
                calcErrorH = hiddenOutput * (1 - hiddenOutput) * outputSum
                #Calculating weight change necessary for hidden-input weight and changing accordingly    
                weightChangeH = self.learnRate * calcErrorH * nodeInput
                self.toHiddenWeights[row][col] += weightChangeH



    #Algorithm intitiates process of learning and building correct weights
    #For each iteration it uses twoarrays, iterates through each tuples in returned arrays, intializes file, runs forward propgoation, and runs backprogpaation
    def learningAlgo(self):
        curDataSet = []
        for i in range(self.numIterate):
            curDataSet = TwoArrays(self.inputSet, self.resultSet).createList()
            for j in range(len(curDataSet)):
                self.initFile(curDataSet[j][0],curDataSet[j][1])
                self.forwardProp()
                self.backProp()

    #Tests validation file and prints results
    def testInputs(self, valInput, valResult):
        for i in range(len(valInput)):
            self.initFile(valInput[i], valResult[i])
            self.forwardProp()
            numResult = valResult[i].index('1')
            #Finds which of the outputNodes has the highest output value 
            maxNum = max(self.outputList, key=lambda node: node.output)
            indexMax = self.outputList.index(maxNum)
            print("Validation line " + str(i) + ": " + "Program determiend the number was " + str(indexMax) + " - The expected number was " + str(numResult))

    #Method used to test one singular inptu vs its expected value 
    def testOne(self, inputString, expectString):
        self.initFile(inputString, expectString)
        self.forwardProp() 

        print("Outputted Values: ")
        for i in range(self.outputSize):
            print(str(i) + ": " + str(self.outputList[i].output))
        print("Expected Values: ")
        for i in range(self.outputSize):
            print(str(i) + ": " + str(self.outputList[i].expect)) 

    l
class Driver(): 
    #intiliazing all values needed for netural netowkr 
    def __init__(self, inputSize,hiddenSize,outputSize,learnRate, numIterate):
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.learnRate = learnRate
        self.numIterate = numIterate
        self.inputDataSet = [] 
        self.resultDataSet = [] 
        self.valInput = []
        self.valResult = []

    
    def setup(self):
        #Reads through file of sample/train data and stores them
        sampleFile = open('SampleData.txt', 'r') 
        sampleLines = sampleFile.readlines() 
        for line in sampleLines:
            if line.strip() != "":
                #Stored into arrays with input and expected results 
                self.inputDataSet.append(line.strip().split(',')[0])
                self.resultDataSet.append(line.strip().split(',')[1])

        #Reads through file of validaiton/train data and stores them
        validaitonFile = open('ValidationData.txt', 'r')
        validationLines = validaitonFile.readlines() 
        for line in validationLines:
            if line.strip() != "":
                #Stored into arrays with input and expected results 
                self.valInput.append(line.strip().split(',')[0])
                self.valResult.append(line.strip().split(',')[1])

        #Creates an object of neural network
        newNN = neuralNetwrok(self.inputSize, self.hiddenSize,self.outputSize, self.learnRate, self.numIterate, self.inputDataSet, self.resultDataSet)
        #Creates all nodes, intializes the weights, and starts algorithm to learn and get correct weights
        newNN.createNodes()
        newNN.initWeights()
        newNN.learningAlgo()
        #Tests the weights with the validation file 
        newNN.testInputs(self.valInput, self.valResult)


class TwoArrays():
    # Read all of the input and resuslet sets into a turple, and add the tuples to an array 
    def __init__(self, inputSet, resultSet):
        self.tupleArray = []
        self.inputSet = inputSet
        self.resultSet = resultSet 

    #Creates input and expected result list and shuffles them to add randomness
    def createList(self):
        for i in range(len(self.inputSet)):
            self.tupleArray.append((self.inputSet[i],self.resultSet[i]))
        random.shuffle(self.tupleArray)
        return self.tupleArray

#Inptu class with value attributed that is ready to be set
class inputNode: 

    def __init__(self, value):
        self.value = value

#Hidden class with blank attirbutes for output and error
class hiddenNode: 
    
    def __init__(self): 
        self.output = None
        self.error = None 


#Output class with blank attirbutes for output and error
class outputNode: 

    def __init__(self):
        self.expect = None 
        self.output = None  
        self.error = None  

#Main method creates object of driver class and intiates setup that runs through all training and testing 
def main():
    neuralStart = Driver(15, 12, 10, 2, 3000)
    neuralStart.setup() 


main() 
