require 'numo/narray'
require_relative 'loss'

class LossCategoricalCrossentropy < Loss
  def forward(y_hat, y)
    n_samples = y_hat.shape[0]

    # Clip data to prevent division by 0
    # Clip both sides to not drag mean towards any value
    y_hat_clipped = y_hat.clip(1e-7, 1 - 1e-7)

    correct_confidences = case y.shape.size
    when 1
      # Probabilities for target values -
      # only if categorical labels
      y_hat_clipped[0...n_samples, y]
    when 2
      # Mask values - only for one-hot encoded labels
    
      (y_hat_clipped * y).sum(axis: 1)
    end
    
    # Losses
    negative_log_likelihoods = - Numo::DFloat::Math.log(correct_confidences)
  end
end
