require_relative 'activation_relu'
require_relative '../03_adding_layers/layer_dense'

require_relative '../lib/nnfs'
NNFS.init

require_relative '../lib/datasets'

# Create dataset
x, y = SpiralData.create(100, 3)
# Create Dense layer with 2 input features and 3 output values
dense1 = LayerDense.new(2, 3)
# Create ReLU activation (to be used with Dense layer):
activation1 = ActivationReLU.new
# Perform a forward pass of our training data through this layer
dense1.forward(x)
# Forward pass through activation func.
# Takes in output from previous layer
activation1.forward(dense1.output)
# Let's see output of the first few samples:
p activation1.output[0...20, true]
