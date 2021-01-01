require_relative '../lib/nnfs'
NNFS.init

require_relative '../lib/datasets'

require_relative '../03_adding_layers/layer_dense'
require_relative '../04_activation_functions/activation_relu'
require_relative '../04_activation_functions/activation_softmax'
require_relative '../05_calculating_loss/loss_categorical_crossentropy'

x, y = SpiralData.create(n_samples: 100, n_classes: 3)

dense1 = LayerDense.new(n_inputs: 2, n_neurons: 3)
activation1 = ActivationReLU.new
dense2 = LayerDense.new(n_inputs: 3, n_neurons: 3)
activation2 = ActivationSoftmax.new

loss_function = LossCategoricalCrossentropy.new

lowest_loss = 9999999
best_dense1_weights = dense1.weights.dup
best_dense1_biases = dense1.biases.dup
best_dense2_weights = dense2.weights.dup
best_dense2_biases = dense2.biases.dup

100000.times do |iteration|
  dense1.weights += 0.05 + Numo::DFloat.new(2, 3).rand_norm
  dense1.biases += 0.05 + Numo::DFloat.new(1, 3).rand_norm
  dense2.weights += 0.05 + Numo::DFloat.new(3, 3).rand_norm
  dense2.biases += 0.05 + Numo::DFloat.new(1, 3).rand_norm

  dense1.forward(x)
  activation1.forward(dense1.output)
  dense2.forward(activation1.output)
  activation2.forward(dense2.output)

  loss = loss_function.calculate(activation2.output, y)
  predictions = activation2.output.argmax(axis: 1)
  accuracy = Numo::DFloat.cast(predictions.eq(y)).mean

  if loss < lowest_loss
    puts "New set of weights found, iteration: #{iteration}, loss: #{loss} accuracy: #{accuracy}"

    best_dense1_weights = dense1.weights.dup
    best_dense1_biases = dense1.biases.dup
    best_dense2_weights = dense2.weights.dup
    best_dense2_biases = dense2.biases.dup
    lowest_loss = loss
  else
    dense1.weights = best_dense1_weights.dup
    dense1.biases = best_dense1_biases.dup
    dense2.weights = best_dense2_weights.dup
    dense2.biases = best_dense2_biases.dup
  end
end
