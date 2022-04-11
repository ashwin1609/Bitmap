import numpy as np
import matplotlib.pyplot as plt

class BitmapNeuralNetwork(object):

    def __init__(self):
    # defining the input size variables
        self.input_size = 45
        self.output_size = 10
        self.hidden_layer = 3

        # defining weights
        self.weight1 = np.random.randn(self.input_size,self.hidden_layer) # 45 X 3
        self.weight2 = np.random.randn(self.hidden_layer,self.output_size) # 3 X 10

        self.learningRate1 = 0.0038
        self.learningRate2 = 0.0069

    # Activation function used in forward propogation
    def sigmoid(self,sig):
        return 1/(1 + np.exp(-sig))

    # Activation function used in backward propogation
    def sigmoidDerivative(self,sig):
        return sig * (1 - sig)

    # Function to transpose the list
    def transpose(self,templist):
        trans = list(map(list, zip(*templist)))
        return trans

    # Function that forward propogates
    # Actual Output is calculated by finding the sigmoid function of the dot product of input layers with the two weights
    # 
    def forwardPropogation(self, dataset):
        self.z1 = np.dot(dataset,self.weight1)
        self.yj = self.sigmoid(self.z1)
        self.yk = np.dot(self.yj,self.weight2)
        actualOutput = self.sigmoid(self.yk)
        return actualOutput

    def backwardPropogation(self,dataset,output,actualOutput):
        
        self.output_error = output - actualOutput
        self.output_delta = self.output_error * self.sigmoidDerivative(actualOutput)

        self.yj_error = self.output_delta.dot(self.transpose(self.weight2))
        self.yj_delta = self.yj_error * self.sigmoidDerivative(self.yj_error)

        # adjusting weights
        self.weight1 += (np.dot(self.transpose(dataset),self.yj_delta)) * self.learningRate1
        self.weight2 += (np.dot(self.transpose(self.yj),self.output_delta)) * self.learningRate2

    def trainModel(self,dataset,output):

        actualOutput = self.forwardPropogation(dataset)
        self.backwardPropogation(dataset,output,actualOutput)

BNN = BitmapNeuralNetwork()

decimal0 = np.array([
[0, 1, 1, 1, 0],
[1, 0, 0, 0, 1], 
[1, 0, 0, 0, 1], 
[1, 0, 0, 0, 1], 
[1, 0, 0, 0, 1], 
[1, 0, 0, 0, 1], 
[1, 0, 0, 0, 1], 
[1, 0, 0, 0, 1], 
[0, 1, 1, 1, 0]])


decimal1 = np.array([
[0, 0, 1, 0, 0],
[0, 1, 1, 0, 0], 
[1, 0, 1, 0, 0], 
[0, 0, 1, 0, 0], 
[0, 0, 1, 0, 0], 
[0, 0, 1, 0, 0], 
[0, 0, 1, 0, 0], 
[0, 0, 1, 0, 0], 
[0, 0, 1, 0, 0]])

decimal2 = np.array([
[0, 1, 1, 1, 0],
[1, 0, 0, 0, 1], 
[0, 0, 0, 0, 1], 
[0, 0, 0, 0, 1], 
[0, 0, 0, 1, 0], 
[0, 0, 1, 0, 0], 
[0, 1, 0, 0, 0], 
[1, 0, 0, 0, 0], 
[1, 1, 1, 1, 1]])

decimal3 = np.array([
[0, 1, 1, 1, 0],
[1, 0, 0, 0, 1], 
[0, 0, 0, 0, 1], 
[0, 0, 0, 0, 1], 
[0, 0, 0, 1, 0], 
[0, 0, 0, 0, 1], 
[0, 0, 0, 0, 1], 
[1, 0, 0, 0, 1], 
[0, 1, 1, 1, 0]])

decimal4 = np.array([
[0, 0, 0, 1, 0],
[0, 0, 1, 1, 0], 
[0, 0, 1, 1, 0], 
[0, 1, 0, 1, 0], 
[0, 1, 0, 1, 0], 
[1, 0, 0, 1, 0], 
[1, 1, 1, 1, 1], 
[0, 0, 0, 1, 0], 
[0, 0, 0, 1, 0]])

decimal5 = np.array([
[1, 1, 1, 1, 1],
[1, 0, 0, 0, 0], 
[1, 0, 0, 0, 0], 
[1, 1, 1, 1, 0], 
[1, 0, 0, 0, 1], 
[0, 0, 0, 0, 1], 
[0, 0, 0, 0, 1], 
[1, 0, 0, 0, 1], 
[0, 1, 1, 1, 0]])

decimal6 = np.array([
[0, 1, 1, 1, 0],
[1, 0, 0, 0, 1], 
[1, 0, 0, 0, 0], 
[1, 0, 0, 0, 0], 
[1, 1, 1, 1, 0], 
[1, 0, 0, 0, 1], 
[1, 0, 0, 0, 1], 
[1, 0, 0, 0, 1], 
[0, 1, 1, 1, 0]])

decimal7 = np.array([
[1, 1, 1, 1, 1],
[0, 0, 0, 0, 1], 
[0, 0, 0, 1, 0], 
[0, 0, 0, 1, 0], 
[0, 0, 1, 0, 0], 
[0, 0, 1, 0, 0], 
[0, 1, 0, 0, 0], 
[0, 1, 0, 0, 0], 
[0, 1, 0, 0, 0]])

decimal8 = np.array([
[0, 1, 1, 1, 0],
[1, 0, 0, 0, 1], 
[1, 0, 0, 0, 1], 
[1, 0, 0, 0, 1], 
[0, 1, 1, 1, 0], 
[1, 0, 0, 0, 1], 
[1, 0, 0, 0, 1], 
[1, 0, 0, 0, 1], 
[0, 1, 1, 1, 0]])

decimal9 = np.array([
[0, 1, 1, 1, 0],
[1, 0, 0, 0, 1], 
[1, 0, 0, 0, 1], 
[1, 0, 0, 0, 1], 
[0, 1, 1, 1, 1], 
[0, 0, 0, 0, 1], 
[0, 0, 0, 0, 1], 
[1, 0, 0, 0, 1], 
[0, 1, 1, 1, 0]])

input1 = decimal0.flatten()
input2 = decimal1.flatten()
input3 = decimal2.flatten()
input4 = decimal3.flatten()
input5 = decimal4.flatten()
input6 = decimal5.flatten()
input7 = decimal6.flatten()
input8 = decimal7.flatten()
input9 = decimal8.flatten()
input10 = decimal9.flatten()

# input dataset
dataset = []
dataset.append(input1)
dataset.append(input2)
dataset.append(input3)
dataset.append(input4)
dataset.append(input5)
dataset.append(input6)
dataset.append(input7)
dataset.append(input8)
dataset.append(input9)
dataset.append(input10)
dataset = np.asarray(dataset) # Dataset is a numpy array

# output dataset
output= []  

for i in range(10):
    temp = []
    for j in range(10):
        if (i == j):
            temp.append(1)
        else:
            temp.append(0)
    output.append(temp)
output = np.asarray(output)

# Number of iterations for training the model
for i in range(2000):
    # For every 20 iterations the current loss is printed
    if (i % 20 == 0):
        print("Current Loss: " + str(np.mean(np.square(output - BNN.forwardPropogation(dataset)))))
    BNN.trainModel(dataset,output)

print("Actual Output : " + str(output))
print("Predicted Output : " + str(BNN.forwardPropogation(dataset)))

print("Loss" + str(np.mean(np.square(output - BNN.forwardPropogation(dataset)))))

xpoints = np.array([0, 100, 500, 1000, 2000])
ypoints = np.array([10, 10**-1, 10**-2, 10**-3, 10**-4])

plt.plot(xpoints, ypoints)
plt.show()










