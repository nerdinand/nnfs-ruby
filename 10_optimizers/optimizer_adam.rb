require 'numo/narray'

class OptimizerAdam
  def initialize(learning_rate: 0.001, decay: 0.0, epsilon: 1e-7, beta_1: 0.9, beta_2: 0.999)
    @learning_rate = learning_rate
    @current_learning_rate = learning_rate
    @decay = decay
    @iterations = 0
    @epsilon = epsilon
    @beta_1 = beta_1
    @beta_2 = beta_2
  end

  attr_reader :current_learning_rate

  def pre_update_params
    return if @decay.zero?

    @current_learning_rate = @learning_rate * ( 1.0 / (1.0 + @decay * @iterations))
  end

  def update_params(layer)
    layer.weight_momentums = @beta_1 * layer.weight_momentums + (1 - @beta_1) * layer.dweights
    layer.bias_momentums = @beta_1 * layer.bias_momentums + (1 - @beta_1) * layer.dbiases

    weight_momentums_corrected = layer.weight_momentums / (1 - @beta_1 ** (@iterations + 1))
    bias_momentums_corrected = layer.bias_momentums / (1 - @beta_1 ** (@iterations + 1))

    layer.weight_cache = @beta_2 * layer.weight_cache + (1 - @beta_2) * layer.dweights ** 2
    layer.bias_cache = @beta_2 * layer.bias_cache + (1 - @beta_2) * layer.dbiases ** 2

    weight_cache_corrected = layer.weight_cache / (1 - @beta_2 ** (@iterations + 1))
    bias_cache_corrected = layer.bias_cache / (1 - @beta_2 ** (@iterations + 1))

    layer.weights += -@current_learning_rate * weight_momentums_corrected / (Numo::DFloat::Math.sqrt(weight_cache_corrected) + @epsilon)
    layer.biases += -@current_learning_rate * bias_momentums_corrected / (Numo::DFloat::Math.sqrt(bias_cache_corrected) + @epsilon)
  end

  def post_update_params
    @iterations += 1
  end
end
