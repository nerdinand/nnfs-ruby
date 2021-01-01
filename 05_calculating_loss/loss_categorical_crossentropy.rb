require 'numo/narray'
require_relative 'loss'

class LossCategoricalCrossentropy < Loss
  def forward(y_pred, y)
    n_samples = y_pred.shape[0]

    # Clip data to prevent division by 0
    # Clip both sides to not drag mean towards any value
    y_pred_clipped = y_pred.clip(1e-7, 1 - 1e-7)

    correct_confidences = case y.shape.size
    when 1
      # Probabilities for target values -
      # only if categorical labels
      y_pred_clipped[0...n_samples, y].diagonal
    when 2
      # Mask values - only for one-hot encoded labels
    
      (y_pred_clipped * y).sum(axis: 1)
    end
    
    # Losses
    negative_log_likelihoods = - Numo::DFloat::Math.log(correct_confidences)
  end
end
