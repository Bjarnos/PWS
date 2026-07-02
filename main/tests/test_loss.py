# type: ignore
import jax.numpy as np
from neural_network import MeanSquaredError, MeanAbsoluteError, CategorialCrossEntropy

mse = MeanSquaredError()
mae = MeanAbsoluteError()
cce = CategorialCrossEntropy(1e-7)

predicted = np.array([[0.1, 0.9], [0.8, 0.2]])
expected = np.array([[0.0, 1.0], [1.0, 0.0]])

print("MSE grad:\n", mse.derivative(predicted, expected))
print("MAE grad:\n", mae.derivative(predicted, expected))
print("CCE grad:\n", cce.derivative(predicted, expected))

