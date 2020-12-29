require_relative '../05_calculating_loss/loss_categorical_crossentropy'

class LossCategoricalCrossentropy
  attr_reader :dinputs
  
  def backward(dvalues, y_true)
    n_samples = dvalues.shape[0]
    n_labels = dvalues.shape[1]

    # If labels are sparse, turn them into one-hot vector
    if y_true.shape.size == 1
      y_true = Numo::DFloat.eye(n_labels)[y_true, true]
    end
    
    # Calculate gradient
    @dinputs = -y_true / dvalues
    # Normalize gradient
    @dinputs = @dinputs / n_samples
  end
end