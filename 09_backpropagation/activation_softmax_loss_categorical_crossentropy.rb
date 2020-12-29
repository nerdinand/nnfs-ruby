require_relative 'activation_softmax'
require_relative 'loss_categorical_crossentropy'

class ActivationSoftmaxLossCategoricalCrossentropy
  attr_reader :dinputs, :output

  def initialize
    @activation = ActivationSoftmax.new
    @loss = LossCategoricalCrossentropy.new
  end

  def forward(inputs, y_true)
    @activation.forward(inputs)
    @output = @activation.output
    @loss.calculate(@output, y_true)
  end

  def backward(dvalues, y_true)
    n_samples = dvalues.shape[0]

    # If labels are one-hot encoded,
    # turn them into discrete values
    if y_true.shape.size == 2
      y_true = y_true.argmax(axis: 1)
    end

    # Copy so we can safely modify
    @dinputs = dvalues.dup
    # Calculate gradient
    @dinputs[0...n_samples, y_true] -= 1
    # Normalize gradient
    @dinputs = @dinputs / n_samples
  end
end