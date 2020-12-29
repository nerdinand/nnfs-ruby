require_relative '../04_activation_functions/activation_relu'

class ActivationReLU
  attr_reader :dinputs
  
  def backward(dvalues)
    # Since we need to modify the original variable,
    # let's make a copy of the values first
    @dinputs = dvalues.dup
    
    # Zero gradient where input values were negative
    @dinputs[@inputs.le(0)] = 0
  end
end
