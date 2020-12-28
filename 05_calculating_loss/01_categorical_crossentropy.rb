require 'numo/narray'
require_relative 'loss_categorical_crossentropy'

softmax_outputs = Numo::DFloat[ 
  [0.7, 0.1, 0.2],
  [0.1, 0.5, 0.4],
  [0.02, 0.9, 0.08]
]
class_targets = Numo::DFloat[
  [1, 0, 0],
  [0, 1, 0],
  [0, 1, 0]
]

loss_function = LossCategoricalCrossentropy.new
loss = loss_function.calculate(softmax_outputs, class_targets)
print(loss)
