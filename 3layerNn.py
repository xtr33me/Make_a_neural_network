from numpy import exp, array, random, dot
class Layer():
    def __init__(self, num_neuron_inputs, num_neurons):
        self.synaptic_weights = 2 * random.random((num_neuron_inputs, num_neurons)) - 1
        self.error = 0.0
        self.adjustment = 0.0

class NeuralNetwork():
    def __init__(self, *layers):
        #Seed the random number generator, so it generates the same numbers
        #every time the program runs
        random.seed(1)

        self.totalLayers = list(layers)
        self.numLayers = len(self.totalLayers)
    
    @staticmethod
    def printWeights(nn):
        print("Current synaptic weights:")
        for i in range(nn.numLayers):
            print(nn.totalLayers[i].synaptic_weights)

    #The sigmoid function, which describes an s shaped curve
    #we pass the weighted sum of the inputs through this function
    #to normalize them between 0 and 1
    def __sigmoid(self, x):
        return 1 /(1 + exp(-x))
    
    #gradient of the sigmoid curve
    def __sigmoid_derivative(self, x):
        return x * (1-x)

    def backpropagate(self, outputs):
        ndx = self.numLayers - 1
        self.totalLayers[ndx].error = outputs - self.totalLayers[ndx].predict
        self.totalLayers[ndx].adjustment = dot(self.totalLayers[ndx-1].predict.T, self.totalLayers[ndx].error * self.__sigmoid_derivative(self.totalLayers[ndx].predict))
        for i in range(ndx-1, -1, -1):
            self.totalLayers[i].error = outputs - self.totalLayers[i].predict
            self.totalLayers[i].adjustment = dot(self.totalLayers[i-1].predict.T, self.totalLayers[i].error * self.__sigmoid_derivative(self.totalLayers[i].predict))
            self.totalLayers[i].synaptic_weights += self.totalLayers[i].adjustment
    
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            #pass the training set through our neural net
            self.predict(training_set_inputs)
            self.backpropagate(training_set_outputs)    
    
    def predict(self, inputs):
        #pass inputs through our neural network (single neuron)
        self.totalLayers[0].predict = self.__sigmoid(dot(inputs, self.totalLayers[0].synaptic_weights))
        for i in range(1, self.numLayers):
            self.totalLayers[i].predict = self.__sigmoid(dot(self.totalLayers[i-1].predict, self.totalLayers[i].synaptic_weights))
        return self.totalLayers[self.numLayers-1].predict


if __name__ == '__main__':
    l1 = Layer(3,3)
    l2 = Layer(3,3)
    l3 = Layer(3,1)
    #init a single neuron NN
    neuralNetwork = NeuralNetwork(l1, l2, l3)

    print ('Random starting synaptic weight:')
    NeuralNetwork.printWeights(neuralNetwork)
    
    #The training set contains 4 examples, each consisting of 3 input values and 1 output
    training_set_inputs = array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
    training_set_outputs = array([[0,1,1,0]]).T

    #Train the neural network using a training set
    #Do it 10000 times and make small adjustments each time
    neuralNetwork.train(training_set_inputs, training_set_outputs, 20000)

    print ("New synaptic weights after training: ")
    NeuralNetwork.printWeights(neuralNetwork)

    #Test the neural network with a new situation
    print ("Considering new situation [1,0,0] -> ?: ")
    print (neuralNetwork.predict(array([1,0,0])))

