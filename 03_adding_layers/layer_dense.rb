require 'numo/narray'

class LayerDense
  WEIGHTS_SCALING_FACTOR = 0.01

  attr_accessor :weights, :biases
  attr_reader :output, :inputs

  def initialize(n_inputs, n_neurons)
    @weights = WEIGHTS_SCALING_FACTOR * Numo::DFloat.new(n_inputs, n_neurons).rand_norm
    @biases = Numo::DFloat.zeros(1, n_neurons)
  end

  def forward(inputs)
    @inputs = inputs
    @output = inputs.dot(@weights) + @biases
  end
end
