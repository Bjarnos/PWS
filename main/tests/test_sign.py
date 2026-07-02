# type: ignore
import numpy as np

predicted = np.array([[0.1, 0.9], [0.8, 0.2]])
expected = np.array([[0.0, 1.0], [1.0, 0.0]])

mae_old = (predicted > expected).astype(np.int32) - (predicted < expected).astype(np.int32)
mae_new = np.sign(predicted - expected)

print("Old:", mae_old)
print("New:", mae_new)
print("Same:", np.array_equal(mae_old, mae_new))
