require 'numo/narray'

require_relative 'layer_dense'
require_relative 'activation_relu'
require_relative 'activation_softmax_loss_categorical_crossentropy'

require_relative '../lib/nnfs'
NNFS.init

require_relative '../lib/datasets'

# Create dataset
x, y = SpiralData.create(n_samples: 100, n_classes: 3)
# Create Dense layer with 2 input features and 3 output values
dense1 = LayerDense.new(n_inputs: 2, n_neurons: 3)
# Create ReLU activation (to be used with Dense layer):
activation1 = ActivationReLU.new
# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values
dense2 = LayerDense.new(n_inputs: 3, n_neurons: 3)

# Create Softmax classifier's combined loss and activation
loss_activation = ActivationSoftmaxLossCategoricalCrossentropy.new

# Make a forward pass of our training data through this layer
dense1.forward(x)
# Make a forward pass through activation function
# it takes the output of first dense layer here
activation1.forward(dense1.output)
# Make a forward pass through second Dense layer
# it takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)
# Perform a forward pass through the activation/loss function
# takes the output of second dense layer here and returns loss
loss = loss_activation.forward(dense2.output, y)
# Let's see output of the first few samples:
p loss_activation.output[0...5, true]

# Print loss value
puts "Loss: #{loss}"

# Calculate accuracy from output of loss_activation and targets
# calculate values along first axis
predictions = loss_activation.output.argmax(axis: 1)
if y.shape.size == 2
  y = y.argmax(axis: 1)
end

accuracy = Numo::DFloat.cast(predictions.eq(y)).mean
# Print accuracy
puts "Accuracy: #{accuracy.inspect}"

# Backward pass
loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

# Print gradients
puts dense1.dweights.inspect
puts dense1.dbiases.inspect
puts dense2.dweights.inspect
puts dense2.dbiases.inspect
