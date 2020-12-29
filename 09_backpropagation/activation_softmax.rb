require 'numo/narray'

require_relative '../04_activation_functions/activation_softmax'

class ActivationSoftmax
  attr_reader :dinputs

  def backward(dvalues)
    @dinputs = Numo::DFloat.new_like(dvalues).allocate

    @output.to_a.zip(dvalues.to_a).each.with_index do |(single_output, single_dvalues), index|
      # Flatten output array
      single_output = Numo::DFloat.cast(single_output)
      
      single_output_reshaped = single_output.reshape(true, 1)
      # Calculate Jacobian matrix of the output
      jacobian_matrix = single_output.diag - single_output_reshaped.dot(single_output_reshaped.transpose)
      # Calculate sample-wise gradient
      # and add it to the array of sample gradients
      @dinputs[index, true] = jacobian_matrix.dot(single_dvalues)
    end
  end
end