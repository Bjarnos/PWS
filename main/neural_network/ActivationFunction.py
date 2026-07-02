# source: https://github.com/jElhamm/Activation-Functions/
import jax.numpy as np

class ActivationFunction:
    def __init__(self):
        pass

    def calculate(self, x: np.ndarray) -> np.ndarray: return np.empty((0, 0)) # pyright: ignore[reportUnknownMemberType]

    def derivative(self, x: np.ndarray) -> np.ndarray: return np.empty((0, 0)) # pyright: ignore[reportUnknownMemberType]

class Linear(ActivationFunction):
    def __init__(self, sharpness: float = 1):
        self.sharpness = sharpness

    def calculate(self, x: np.ndarray) -> np.ndarray:
        return x
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x) # pyright: ignore[reportUnknownMemberType]
    
class ReLU(ActivationFunction):
    def __init__(self, sharpness: float = 1):
        self.sharpness = sharpness

    def calculate(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x * self.sharpness)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, self.sharpness, 0)
    
class LeakyReLU(ActivationFunction):
    def __init__(self, slope: float = 0.01, sharpness: float = 1):
        self.slope = slope
        self.sharpness = sharpness
        
    def calculate(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(self.slope * x, self.sharpness * x)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, self.sharpness, self.slope)

class Softplus(ActivationFunction):
    def calculate(self, x: np.ndarray) -> np.ndarray:
        return np.log(1 + np.exp(x))
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
class ELU(ActivationFunction):
    def __init__(self, saturation: float = 1):
        self.saturation = saturation
        
    def calculate(self, x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0, x, self.saturation * (np.exp(x) - 1))
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0, 1, self.calculate(x) + self.saturation)

class SELU(ActivationFunction):
    def __init__(self, saturation: float = 1, scale: float = 1):
        self.saturation = saturation
        self.scale = scale
        
    def calculate(self, x: np.ndarray) -> np.ndarray:
        return self.scale * np.where(x >= 0, x, self.saturation * (np.exp(x) - 1))
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return self.scale * np.where(x >= 0, 1, self.saturation * np.exp(x))

class GELU(ActivationFunction):
    def calculate(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
    
    def derivative(self, x: np.ndarray) -> np.ndarray: 
        cdf = 0.5 * (1 + np.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))) + (x / np.sqrt(2))))
        return 0.5 * (1 + cdf + x * (1 - cdf))
    
class Gaussian(ActivationFunction):
    def calculate(self, x: np.ndarray) -> np.ndarray:
        return np.exp(-x**2)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return -2 * x * np.exp(-x**2)

class Sigmoid(ActivationFunction):
    def calculate(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        v = self.calculate(x)
        return v * (1 - v)
    
class Softsign(ActivationFunction):
    def calculate(self, x: np.ndarray) -> np.ndarray:
        return x / (np.abs(x) + 1)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 / np.square(np.abs(x) + 1)
    
class Swish(ActivationFunction):
    def calculate(self, x: np.ndarray) -> np.ndarray:
        return x * (1 / (1 + np.exp(-x)))
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        v = self.calculate(x)
        return v + (1 - v) / (1 + np.exp(-x))
    
class Tanh(ActivationFunction):
    def calculate(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.square(self.calculate(x))

class Softmax(ActivationFunction):
    def calculate(self, x: np.ndarray) -> np.ndarray:
        shifted_x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(shifted_x)
        sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
        return exp_x / sum_exp_x
