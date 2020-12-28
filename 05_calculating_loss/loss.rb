class Loss
  def calculate(y_hat, y)
    forward(y_hat, y).mean
  end
end
