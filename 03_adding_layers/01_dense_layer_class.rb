require_relative 'layer_dense'

require_relative '../lib/nnfs'
NNFS.init

require_relative '../lib/datasets'

# Create dataset
x, y = SpiralData.create(100, 3)
# Create Dense layer with 2 input features and 3 output values
dense1 = LayerDense.new(2, 3)
# Perform a forward pass of our training data through this layer
dense1.forward(x)
# Let's see output of the first few samples:
p dense1.output[0...5, true]
