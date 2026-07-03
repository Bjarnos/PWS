"""
Some functions below have a `sharpness` parameter. You can give this
a value when instantiating the class to change how effective the
function is.

Used source: https://github.com/jElhamm/Activation-Functions/
"""

import jax.numpy as np

class ActivationFunction:
    """
    The parent class for all activation functions.

    <em>You should not use this class directly, but rather
    one of the child classes. It is only exported to use
    for typing.</em>
    """

    def __init__(self):
        pass

    def calculate(self, x: np.ndarray) -> np.ndarray:
        "@private"
        return np.empty((0, 0)) # pyright: ignore[reportUnknownMemberType]

    def derivative(self, x: np.ndarray) -> np.ndarray:
        "@private"
        return np.empty((0, 0)) # pyright: ignore[reportUnknownMemberType]

class Linear(ActivationFunction):
    """
    A simple activation function which returns
    the input value unchanged
    """

    def __init__(self, sharpness: float = 1):
        self.sharpness = sharpness

    def calculate(self, x: np.ndarray) -> np.ndarray:
        r"""$$\text{Linear}(x)=x$$"""
        return x * self.sharpness
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        r"""$$\text{Linear}'(x)=1$$"""
        return np.ones_like(x) * self.sharpness # pyright: ignore[reportUnknownMemberType]
    
class ReLU(ActivationFunction):
    """
    <em>Rectified Linear Unit</em><br>
    An activation function which returns the
    non-negative part of its input
    """

    def __init__(self, sharpness: float = 1):
        self.sharpness = sharpness

    def calculate(self, x: np.ndarray) -> np.ndarray:
        r"""$$\text{ReLU}(x)=\begin{cases}x, & \text{if } x > 0 \\\\0, & \text{if } x < 0\end{cases}$$"""
        return np.maximum(0, x * self.sharpness)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        r"""$$\text{ReLU}'(x) = \begin{cases}1, & \text{if } x > 0 \\\\0, & \text{if } x < 0\end{cases}$$"""
        return np.where(x > 0, self.sharpness, 0)
    
class LeakyReLU(ActivationFunction):
    """
    <em>Leaky Rectified Linear Unit</em><br>
    An activation function that allows a small,
    non-zero gradient when the input is negative
    """

    def __init__(self, slope: float = 0.01, sharpness: float = 1):
        self.slope = slope
        self.sharpness = sharpness
        
    def calculate(self, x: np.ndarray) -> np.ndarray:
        r'''$$\text{LeakyReLU}(x)=\begin{cases}x, & \text{if } x > 0 \\\\\alpha{x}, & \text{if } x < 0\end{cases}$$'''
        return np.maximum(self.slope * x, self.sharpness * x)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        r'''$$\text{LeakyReLU}'(x)=\begin{cases}1, & \text{if } x > 0 \\\\\alpha, & \text{if } x < 0\end{cases}$$'''
        return np.where(x > 0, self.sharpness, self.slope)

class Softplus(ActivationFunction):
    """
    A smooth approximation of the ReLU function
    """

    def calculate(self, x: np.ndarray) -> np.ndarray:
        r'''$$\text{Softplus}(x)=\ln(1+e^x)$$'''
        return np.log(1 + np.exp(x))
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        r'''$$\text{Softplus}'(x)=\frac{1}{1+e^x}$$'''
        return 1 / (1 + np.exp(-x))
    
class ELU(ActivationFunction):
    """
    A variant of the ReLU function which allows
    for a negative and near-zero input
    """

    def __init__(self, sharpness: float = 1):
        self.sharpness = sharpness
        
    def calculate(self, x: np.ndarray) -> np.ndarray:
        r'''$$\text{ELU}(x)=\begin{cases}x, & \text{if } x > 0 \\\\e^x-1, & \text{if } x < 0\end{cases}$$'''
        return np.where(x >= 0, x, self.sharpness * (np.exp(x) - 1))
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        r'''$$\text{ELU}'(x)=\begin{cases}1, & \text{if } x > 0 \\\\\text{ELU}(x) + 1, & \text{if } x < 0\end{cases}$$'''
        return np.where(x >= 0, 1, self.calculate(x) + self.sharpness)

class SELU(ActivationFunction):
    """
    A variant of the ReLU function which keeps the
    output normalized for more stability
    """

    def __init__(self, sharpness: float = 1, scale: float = 1):
        self.sharpness = sharpness
        self.scale = scale
        
    def calculate(self, x: np.ndarray) -> np.ndarray:
        r'''$$\text{SELU}(x)=\begin{cases}\lambda{x}, & \text{if } x > 0 \\\\\lambda{e^x-1}, & \text{if } x < 0\end{cases}$$'''
        return self.scale * np.where(x >= 0, x, self.sharpness * (np.exp(x) - 1))
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        r'''$$\text{SELU}'(x)=\begin{cases}\lambda, & \text{if } x > 0 \\\\\lambda{e^x}, & \text{if } x < 0\end{cases}$$'''
        return self.scale * np.where(x >= 0, 1, self.sharpness * np.exp(x))

class GELU(ActivationFunction):
    """
    An activation function which returns the
    probability that a Gaussian random variable
    is less than or equal to x
    """

    def calculate(self, x: np.ndarray) -> np.ndarray:
        r'''$$\text{GELU}(x)=0.5x(1+\text{tanh}(\sqrt{\frac{2}{\pi}}(x+0.044715x^3)))$$'''
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
    
    def derivative(self, x: np.ndarray) -> np.ndarray: 
        r'''$$\text{GELU}'(x)=0.5(1+0.5(1 + \text{tanh}(\sqrt{\frac{2}{\pi}}(x+0.044715x^3) + (\frac{x}{\sqrt{2}})))+x(1-0.5(1 + \text{tanh}(\sqrt{\frac{2}{\pi}}(x+0.044715x^3) + (\frac{x}{\sqrt{2}})))))$$'''
        cdf = 0.5 * (1 + np.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))) + (x / np.sqrt(2))))
        return 0.5 * (1 + cdf + x * (1 - cdf))
    
class Gaussian(ActivationFunction):
    """
    An activation function that maps inputs to outputs
    based on a Gaussian distribution
    """

    def calculate(self, x: np.ndarray) -> np.ndarray:
        r'''$$\text{Gaussian}(x)={e^{-x^2}}$$'''
        return np.exp(-x**2)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        r'''$$\text{Gaussian}'(x)=-2x\cdot{e^{-x^2}}$$'''
        return -2 * x * np.exp(-x**2)

class Sigmoid(ActivationFunction):
    """
    An activation function that maps inputs to a
    value between 0 and 1
    """

    def calculate(self, x: np.ndarray) -> np.ndarray:
        r'''$$\text{Sigmoid}(x)=\frac{1}{1+e^{-x}}$$'''
        return 1 / (1 + np.exp(-x))
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        r'''$$\text{Sigmoid}'(x)=\text{Sigmoid}(x)(1-\text{Sigmoid}(x))$$'''
        v = self.calculate(x)
        return v * (1 - v)
    
class Softsign(ActivationFunction):
    """
    An activation function that maps inputs to a
    value between -1 and 1
    """

    def calculate(self, x: np.ndarray) -> np.ndarray:
        r'''$$\text{Softsign}(x)=\frac{x}{|{x}|+1}$$'''
        return x / (np.abs(x) + 1)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        r'''$$\text{Softsign}'(x)=\frac{1}{(|{x}|+1)^2}$$'''
        return 1 / np.square(np.abs(x) + 1)
    
class Swish(ActivationFunction):
    """
    An activation function used to interpolate between
    a linear function and the ReLU function
    """

    def calculate(self, x: np.ndarray) -> np.ndarray:
        r'''$$\text{Swish}(x)=\frac{x}{1+e^{-x}}$$'''
        return x / (1 + np.exp(-x))
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        r'''$$\text{Swish}'(x)=\frac{\text{Swish}(x)+(1-\text{Swish}(x))}{1+e^{-x}}$$'''
        v = self.calculate(x)
        return v + (1 - v) / (1 + np.exp(-x))
    
class Tanh(ActivationFunction):
    """
    An activation function that maps inputs to a
    value between -1 and 1
    """

    def calculate(self, x: np.ndarray) -> np.ndarray:
        r'''$$\text{Tanh}(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}$$'''
        return np.tanh(x)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        r'''$$\text{Tanh}'(x)=1-\text{Tanh}(x)^2$$'''
        return 1 - np.square(self.calculate(x))

class Softmax(ActivationFunction):
    """
    An activation function that maps a vector of inputs
    to values between 0 and 1 with a total sum of 1
    """

    def calculate(self, x: np.ndarray) -> np.ndarray:
        r'''$$\text{Softmax}(x_{i})=\frac{e^{x_{i}}}{\sum_{j=1}^{n}e^{x_{j}}}$$'''
        shifted_x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(shifted_x)
        sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
        return exp_x / sum_exp_x

__all__ = [
    "ActivationFunction", "Linear", "ReLU", "LeakyReLU",
    "Softplus", "ELU", "SELU", "GELU", "Gaussian",
    "Sigmoid", "Softsign", "Swish", "Tanh", "Softmax"
    ]