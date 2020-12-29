require 'numo/narray'

require_relative 'activation_softmax_loss_categorical_crossentropy'
require_relative 'activation_softmax'
require_relative 'loss_categorical_crossentropy'

softmax_outputs = Numo::DFloat[
  [0.7, 0.1, 0.2],
  [0.1, 0.5, 0.4],
  [0.02, 0.9, 0.08]
]

class_targets = Numo::DFloat[0, 1, 1]

softmax_loss = ActivationSoftmaxLossCategoricalCrossentropy.new
softmax_loss.backward(softmax_outputs, class_targets)
dvalues1 = softmax_loss.dinputs

activation = ActivationSoftmax.new
activation.output = softmax_outputs
loss = LossCategoricalCrossentropy.new
loss.backward(softmax_outputs, class_targets)
activation.backward(loss.dinputs)
dvalues2 = activation.dinputs

puts 'Gradients: combined loss and activation:'
p dvalues1
puts 'Gradients: separate loss and activation:'
p dvalues2
