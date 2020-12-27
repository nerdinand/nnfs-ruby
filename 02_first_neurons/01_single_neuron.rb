require 'numo/narray'

inputs = Numo::NArray[1.0, 2.0, 3.0, 2.5]
weights = Numo::NArray[0.2, 0.8, -0.5, 1.0]
bias = 2.0

outputs = weights.dot(inputs) + bias

p outputs
