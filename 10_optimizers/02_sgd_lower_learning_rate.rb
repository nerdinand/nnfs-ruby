require 'numo/narray'

require_relative '../09_backpropagation/layer_dense'
require_relative '../09_backpropagation/activation_relu'
require_relative '../09_backpropagation/activation_softmax_loss_categorical_crossentropy'

require_relative 'optimizer_sgd'

require_relative '../lib/nnfs'
NNFS.init

require_relative '../lib/datasets'

# Create dataset
x, y = SpiralData.create(n_samples: 100, n_classes: 3)
# Create Dense layer with 2 input features and 64 output values
dense1 = LayerDense.new(n_inputs: 2, n_neurons: 64)
# Create ReLU activation (to be used with Dense layer):
activation1 = ActivationReLU.new
# Create second Dense layer with 64 input features (as we take output
# of previous layer here) and 3 output values (output values)
dense2 = LayerDense.new(n_inputs: 64, n_neurons: 3)
# Create Softmax classifier's combined loss and activation
loss_activation = ActivationSoftmaxLossCategoricalCrossentropy.new
# Create optimizer
optimizer = OptimizerSGD.new(learning_rate: 0.85)

# Train in loop
10001.times do |epoch|
  # Perform a forward pass of our training data through this layer
  dense1.forward(x)
  # Perform a forward pass through activation function
  # takes the output of first dense layer here
  activation1.forward(dense1.output)
  # Perform a forward pass through second Dense layer
  # takes outputs of activation function of first layer as inputs
  dense2.forward(activation1.output)
  # Perform a forward pass through the activation/loss function
  # takes the output of second dense layer here and returns loss
  loss = loss_activation.forward(dense2.output, y)  

  # Calculate accuracy from output of loss_activation and targets
  # calculate values along first axis
  predictions = loss_activation.output.argmax(axis: 1)
  if y.shape.size == 2
    y = y.argmax(axis: 1)
  end

  accuracy = Numo::DFloat.cast(predictions.eq(y)).mean

  if (epoch % 100).zero?
    puts "epoch: #{epoch}, acc: #{accuracy.round(3)}, loss: #{loss.round(3)}"
  end

  # Backward pass
  loss_activation.backward(loss_activation.output, y)
  dense2.backward(loss_activation.dinputs)
  activation1.backward(dense2.dinputs)
  dense1.backward(activation1.dinputs)
  
  # Update weights and biases
  optimizer.update_params(dense1)
  optimizer.update_params(dense2)
end
