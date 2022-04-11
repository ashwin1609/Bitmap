import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork(object):

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

    def sigmoid(self,sig):
        return 1/(1 + np.exp(-sig))

    def sigmoidDerivative(self,sig):
        return sig * (1 - sig)

    def transpose(self,templist):
        trans = list(map(list, zip(*templist)))
        return trans

    def forwardPropogation(self, dataset):
        self.z = np.dot(dataset,self.weight1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2,self.weight2)
        actualOutput = self.sigmoid(self.z3)
        return actualOutput

    def backwardPropogation(self,dataset,output,actualOutput):
        
        self.output_error = output - actualOutput
        self.output_delta = self.output_error * self.sigmoidDerivative(actualOutput)

        self.z2_error = self.output_delta.dot(self.transpose(self.weight2))
        self.z2_delta = self.z2_error * self.sigmoidDerivative(self.z2_error)

        # adjusting weights
        self.weight1 += (np.dot(self.transpose(dataset),self.z2_delta)) * self.learningRate1
        self.weight2 += (np.dot(self.transpose(self.z2),self.output_delta)) * self.learningRate2

    def train(self,dataset,output):

        actualOutput = self.forwardPropogation(dataset)
        self.backwardPropogation(dataset,output,actualOutput)


NN = NeuralNetwork()

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

#print(" Simply printint the O/P" + str(output))


for i in range(2000):
    if (i % 20 == 0):
        print("Loss: " + str(np.mean(np.square(output - NN.forwardPropogation(dataset)))))
    NN.train(dataset,output)

print("Actual Output : " + str(output))
print("Predicted Output : " + str(NN.forwardPropogation(dataset)))

print("Loss" + str(np.mean(np.square(output - NN.forwardPropogation(dataset)))))


xpoints = np.array([0, 100, 500, 1000, 2000])
ypoints = np.array([10, 10**1, 10**2, 10**3, 10**4])

plt.plot(xpoints, ypoints)
plt.show()