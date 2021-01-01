class Loss
  def calculate(y_pred, y)
    forward(y_pred, y).mean
  end
end
