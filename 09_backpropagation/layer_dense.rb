require_relative '../03_adding_layers/layer_dense'

class LayerDense
  attr_reader :dinputs, :dweights, :dbiases

  # dvalues are the derivatives of the following layer
  def backward(dvalues)
    # gradients on parameters
    @dweights = @inputs.transpose.dot(dvalues)
    @dbiases = dvalues.sum(axis: 0, keepdims: true)

    # gradient on values
    @dinputs = dvalues.dot(@weights.transpose)
  end
end
