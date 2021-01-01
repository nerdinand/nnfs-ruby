class OptimizerSGD
  def initialize(learning_rate=1.0)
    @learning_rate = learning_rate
  end

  def update_params(layer)
    layer.weights += -@learning_rate * layer.dweights
    layer.biases += -@learning_rate * layer.dbiases
  end
end
