"""
We recommend to turn on "Show Advanced APIs" to see all formulas.

TODO: move all formules to the base class so they are always shown.

Used source: https://www.mdpi.com/2227-7390/11/11/2466
"""

import jax.numpy as np

class Optimizer:
    """
    The parent class for all optimizers.

    <em>You should not use this class directly, but rather
    one of the child classes. It is only exported to use
    for typing.</em>
    """

    def calculate(self, param: np.ndarray, state: dict[str, np.ndarray], grad: np.ndarray) -> np.ndarray:
        "The method to calculate this optimizer.<br>@advanced"
        return param

class SGD(Optimizer):
    """
    <em>Stochastic Gradient Descent</em><br>
    An optimizer which updates the model parameters based on
    the gradient.
    """

    def __init__(self, learning_rate: float = 0.5):
        self.learning_rate = learning_rate
        "Reference to the set learning rate<br>@advanced"

    def calculate(self, param: np.ndarray, state: dict[str, np.ndarray], grad: np.ndarray) -> np.ndarray:
        r"""$$\theta_{t+1}=\theta_t-\alpha g_t$$<br>@advanced"""
        
        return param - self.learning_rate * grad

class SGDM(Optimizer):
    """
    <em>Stochastic Gradient Descent with Momentum</em><br>
    An optimizer based on SGD with a momentum to smooth
    out the updates.
    """

    def __init__(self, learning_rate: float = 1.0, momentum: float = 0.9):
        self.learning_rate = learning_rate
        "Reference to the set learning rate<br>@advanced"
        self.momentum = momentum
        "Reference to the set momentum<br>@advanced"

    def calculate(self, param: np.ndarray, state: dict[str, np.ndarray], grad: np.ndarray) -> np.ndarray:
        r"""$$m_{t+1}=\mu m_t + (1-\mu)g_t, \\ \theta_{t+1}=\theta_t-\alpha m_{t+1}$$<br>@advanced"""

        if "momentum" not in state:
            state["momentum"] = np.zeros_like(param)

        state["momentum"] = (1 - self.momentum) * grad + self.momentum * state["momentum"]
        return param - self.learning_rate * state["momentum"]

class AdaGrad(Optimizer):
    """
    <em>Adaptive Gradients</em><br>
    An optimizer which adapts the learning rate for parameters
    based on its historical gradients.
    """

    def __init__(self, learning_rate: float = 0.05, epsilon: float = 1e-9):
        self.learning_rate = learning_rate
        "Reference to the set learning rate<br>@advanced"
        self.epsilon = epsilon
        "Reference to the set epsilon value<br>@advanced"

    def calculate(self, param: np.ndarray, state: dict[str, np.ndarray], grad: np.ndarray) -> np.ndarray:
        r"""$$G_{t+1}=G_t+g_t^2, \\ \theta_{t+1}=\theta_t-\frac{\alpha}{\sqrt{G_{t+1}}+\epsilon}g_t$$<br>@advanced"""
        
        if "acc" not in state:
            state["acc"] = np.zeros_like(param)

        state["acc"] = state["acc"] + np.square(grad)
        lr = self.learning_rate / np.sqrt(state["acc"] + self.epsilon)
        return param - lr * grad
    
class RMSprop(Optimizer):
    """
    <em>Root Mean Square Propagation</em><br>
    An adapted version of AdaGrad without rapidly shrinking
    learning rates.
    """

    def __init__(self, learning_rate: float = 0.005, decay: float = 0.9, epsilon: float = 1e-9):
        self.learning_rate = learning_rate
        "Reference to the set learning rate<br>@advanced"
        self.decay = decay
        "Reference to the set decay<br>@advanced"
        self.epsilon = epsilon
        "Reference to the set epsilon value<br>@advanced"

    def calculate(self, param: np.ndarray, state: dict[str, np.ndarray], grad: np.ndarray) -> np.ndarray:
        r"""$$v_t=\gamma v_{t-1}+(1-\gamma)g^2_t, \\ \theta_{t+1}=\theta_t - \frac{\alpha}{\sqrt{v_t+\epsilon}}g_t$$<br>@advanced"""
        
        if "acc" not in state:
            state["acc"] = np.zeros_like(param)

        state["acc"] = self.decay * state["acc"] + (1 - self.decay) * np.square(grad)
        lr = self.learning_rate / np.sqrt(state["acc"] + self.epsilon)
        return param - lr * grad
    
class Adam(Optimizer):
    """
    <em>Adaptive Moment Estimation</em><br>
    A combined version of the Momentum and RMSprop optimizers.
    """

    def __init__(self, learning_rate: float = 0.01, decay_ma: float = 0.9, decay_sq: float = 0.999, epsilon: float = 1e-9):
        self.learning_rate = learning_rate
        "Reference to the set learning rate<br>@advanced"
        self.decay_ma = decay_ma
        "Reference to the set decay_ma<br>@advanced"
        self.decay_sq = decay_sq
        "Reference to the set decay_sq<br>@advanced"
        self.epsilon = epsilon
        "Reference to the set epsilon value<br>@advanced"

    def calculate(self, param: np.ndarray, state: dict[str, np.ndarray], grad: np.ndarray) -> np.ndarray:
        r"""$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t, \\ v_t=\beta_2v_{t-1}+(1-\beta_2)g^2_t, \\ \hat{m_t}=\frac{m_t}{(1-\beta_1)}, \\ \hat{v_t}=\frac{v_t}{(1-\beta_2)}, \\ \theta_t=\theta_{t-1}-\alpha\frac{\hat{m_t}}{\sqrt{\hat{v_t}}+\epsilon}$$<br>@advanced"""
        
        if "momentum" not in state:
            state["momentum"] = np.zeros_like(param)
            state["var"] = np.zeros_like(param)

        state["momentum"] = self.decay_ma * state["momentum"] + (1 - self.decay_ma) * grad

        state["var"] = self.decay_sq * state["var"] + (1 - self.decay_sq) * np.square(grad)
        
        momentum_corrected = state["momentum"] / (1 - self.decay_ma)

        var_corrected = state["var"] / (1 - self.decay_sq)

        return param - momentum_corrected / (np.sqrt(var_corrected) + self.epsilon) * self.learning_rate
    
__all__ = ["Optimizer", "SGD", "SGDM", "AdaGrad", "RMSprop", "Adam"]
