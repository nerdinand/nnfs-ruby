require 'numo/narray'

class LayerDense
  WEIGHTS_SCALING_FACTOR = 0.01

  attr_accessor :weights, :biases
  attr_reader :output, :inputs
  attr_accessor :weight_momentums, :bias_momentums
  attr_accessor :weight_cache, :bias_cache

  def initialize(n_inputs:, n_neurons:)
    @weights = WEIGHTS_SCALING_FACTOR * Numo::DFloat.new(n_inputs, n_neurons).rand_norm
    @biases = Numo::DFloat.zeros(1, n_neurons)
    @weight_momentums = @weights.new_zeros
    @bias_momentums = @biases.new_zeros
    @weight_cache = @weights.new_zeros
    @bias_cache = @biases.new_zeros
  end

  def forward(inputs)
    @inputs = inputs
    @output = inputs.dot(@weights) + @biases
  end
end
