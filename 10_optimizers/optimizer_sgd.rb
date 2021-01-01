class OptimizerSGD
  def initialize(learning_rate: 1.0, decay: 0.0, momentum: 0.0)
    @learning_rate = learning_rate
    @current_learning_rate = learning_rate
    @decay = decay
    @momentum = momentum
    @iterations = 0
  end

  attr_reader :current_learning_rate

  def pre_update_params
    return if @decay.zero?

    @current_learning_rate = @learning_rate * ( 1.0 / (1.0 + @decay * @iterations))
  end

  def update_params(layer)
    if @momentum.zero?
      weight_updates = -@learning_rate * layer.dweights
      bias_updates = -@learning_rate * layer.dbiases
    else
      weight_updates = @momentum * layer.weight_momentums - @current_learning_rate * layer.dweights
      layer.weight_momentums = weight_updates
      bias_updates = @momentum * layer.bias_momentums - @current_learning_rate * layer.dbiases
      layer.bias_momentums = bias_updates
    end

    layer.weights += weight_updates
    layer.biases += bias_updates
  end

  def post_update_params
    @iterations += 1
  end
end
