from numpy import exp, array, random, dot

class NeuralNetwork():
    def __init__(self):
        #Seed the random number generator, so it generates the same numbers
        #every time the program runs
        random.seed(1)

        #Model a single neuron, with 3 input connections and 1 output connectn.
        #Assign random weights to a 3 x 1 matrix, with the values in the range -1 to 1
        #and mean 0
        self.synaptic_weights = 2 * random.random((3,1)) - 1
    
    #The sigmoid function, which describes an s shaped curve
    #we pass the weighted sum of the inputs through this function
    #to normalize them between 0 and 1
    def __sigmoid(self, x):
        return 1 /(1 + exp(-x))
    
    #gradient of the sigmoid curve
    def __sigmoid_derivative(self, x):
        return x * (1-x)
    
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        #for iteration in xrange(number_of_training_iterations):
        #Had to change xrange to range below since using python 3
        for iteration in range(number_of_training_iterations):
            #pass the training set through our neural net
            output = self.predict(training_set_inputs)

            #calculate the error
            error = training_set_outputs - output

            #multiply the error by the input and again by the gradient of the sigmoid curve
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            #adjust the weights
            self.synaptic_weights += adjustment
    
    def predict(self, inputs):
        #pass inputs through our neural network (single neuron)
        return self.__sigmoid(dot(inputs, self.synaptic_weights))

if __name__ == '__main__':
    #init a single neuron NN
    neuralNetwork = NeuralNetwork()

    print ('Random starting synaptic weight:')
    print (neuralNetwork.synaptic_weights)

    #The training set contains 4 examples, each consisting of 3 input values and 1 output
    training_set_inputs = array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
    training_set_outputs = array([[0,1,1,0]]).T

    #Train the neural network using a training set
    #Do it 10000 times and make small adjustments each time
    neuralNetwork.train(training_set_inputs, training_set_outputs, 10000)

    print ("New synaptic weights after training: ")
    print (neuralNetwork.synaptic_weights)

    #Test the neural network with a new situation
    print ("Considering new situation [1,0,0] -> ?: ")
    print (neuralNetwork.predict(array([1,0,0])))

