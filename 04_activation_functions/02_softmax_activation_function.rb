require_relative 'activation_softmax'
require_relative 'activation_relu'
require_relative '../03_adding_layers/layer_dense'

require_relative '../lib/nnfs'
NNFS.init

require_relative '../lib/datasets'

softmax = ActivationSoftmax.new
softmax.forward(Numo::NArray[[1, 2, 3]])
p softmax.output


# Create dataset
x, y = SpiralData.create(n_samples: 100, n_classes: 3)
# Create Dense layer with 2 input features and 3 output values
dense1 = LayerDense.new(n_inputs: 2, n_neurons: 3)
# Create ReLU activation (to be used with Dense layer):
activation1 = ActivationReLU.new
# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values
dense2 = LayerDense.new(n_inputs: 3, n_neurons: 3)
# Create Softmax activation (to be used with Dense layer):
activation2 = ActivationSoftmax.new
# Make a forward pass of our training data through this layer
dense1.forward(x)
# Make a forward pass through activation function
# it takes the output of first dense layer here
activation1.forward(dense1.output)
# Make a forward pass through second Dense layer
# it takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)
# Make a forward pass through activation function
# it takes the output of second dense layer here
activation2.forward(dense2.output)
# Let's see output of the first few samples:
p activation2.output[0...5, true]
