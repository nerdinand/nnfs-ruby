require 'numo/narray'

class ActivationReLU
  attr_reader :output

  def forward(inputs)
    @output = Numo::DFloat.maximum(0, inputs)
  end
end

