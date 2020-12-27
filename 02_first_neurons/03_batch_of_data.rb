require 'numo/narray'

inputs = Numo::NArray[
    [1.0, 2.0, 3.0, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]

weights = Numo::NArray[
    [0.2, 0.8, -0.5, 1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]

bias = Numo::NArray[2.0, 3.0, 0.5]

outputs = inputs.dot(weights.transpose) + bias

p outputs
