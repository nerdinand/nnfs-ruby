require 'numo/narray'

class ActivationSoftmax
  attr_reader :output

  def forward(inputs)
    exp_values = Numo::DFloat::Math.exp(inputs - inputs.max(axis: 1, keepdims: true))
    @output = exp_values / exp_values.sum(axis: 1, keepdims: true)
  end
end
