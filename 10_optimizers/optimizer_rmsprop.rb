require 'numo/narray'

class OptimizerRMSProp
  def initialize(learning_rate: 0.001, decay: 0.0, epsilon: 1e-7, rho: 0.9)
    @learning_rate = learning_rate
    @current_learning_rate = learning_rate
    @decay = decay
    @iterations = 0
    @epsilon = epsilon
    @rho = rho
  end

  attr_reader :current_learning_rate

  def pre_update_params
    return if @decay.zero?

    @current_learning_rate = @learning_rate * ( 1.0 / (1.0 + @decay * @iterations))
  end

  def update_params(layer)
    layer.weight_cache = @rho * layer.weight_cache + (1 - @rho) * layer.dweights ** 2
    layer.bias_cache = @rho * layer.bias_cache + (1 - @rho) * layer.dbiases ** 2

    layer.weights += -@current_learning_rate * layer.dweights / (Numo::DFloat::Math.sqrt(layer.weight_cache) + @epsilon)
    layer.biases += -@current_learning_rate * layer.dbiases / (Numo::DFloat::Math.sqrt(layer.bias_cache) + @epsilon)
  end

  def post_update_params
    @iterations += 1
  end
end
