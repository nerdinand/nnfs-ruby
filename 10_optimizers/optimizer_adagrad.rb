require 'numo/narray'

class OptimizerAdagrad
  def initialize(learning_rate: 1.0, decay: 0.0, epsilon: 1e-7)
    @learning_rate = learning_rate
    @current_learning_rate = learning_rate
    @decay = decay
    @iterations = 0
    @epsilon = epsilon
  end

  attr_reader :current_learning_rate

  def pre_update_params
    return if @decay.zero?

    @current_learning_rate = @learning_rate * ( 1.0 / (1.0 + @decay * @iterations))
  end

  def update_params(layer)
    layer.weight_cache += layer.dweights ** 2
    layer.bias_cache += layer.dbiases ** 2

    layer.weights += -@learning_rate * layer.dweights / (Numo::DFloat::Math.sqrt(layer.weight_cache) + @epsilon)
    layer.biases += -@learning_rate * layer.dbiases / (Numo::DFloat::Math.sqrt(layer.bias_cache) + @epsilon)
  end

  def post_update_params
    @iterations += 1
  end
end