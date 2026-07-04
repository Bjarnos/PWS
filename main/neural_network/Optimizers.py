import jax.numpy as np
# source: https://www.mdpi.com/2227-7390/11/11/2466

class Optimizer:
    """
    The parent class for all optimizers.

    <em>You should not use this class directly, but rather
    one of the child classes. It is only exported to use
    for typing.</em>
    """

    def calculate(self, w_gradient: np.ndarray, b_gradient: np.ndarray, w_var: np.ndarray, b_var: np.ndarray, 
                  acc_w_grad: np.ndarray, acc_b_grad: np.ndarray, weights: np.ndarray, biases: np.ndarray, 
                  w_momentum: np.ndarray, b_momentum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        "@private"
        return (weights, biases)

class SGD(Optimizer):
    """
    <em>Stochastic Gradient Descent</em><br>
    An optimizer which updates the model parameters based on
    the gradient.
    """

    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate

    def calculate(self, w_gradient: np.ndarray, b_gradient: np.ndarray, w_var: np.ndarray, b_var: np.ndarray, 
                  acc_w_grad: np.ndarray, acc_b_grad: np.ndarray, weights: np.ndarray, biases: np.ndarray, 
                  w_momentum: np.ndarray, b_momentum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r'''$$\theta_{t+1}=\theta_t-\alpha g_t$$'''
        
        return (
            weights - self.learning_rate * w_gradient,
            biases - self.learning_rate * b_gradient
        )

class SGDM(Optimizer):
    """
    <em>Stochastic Gradient Descent with Momentum</em><br>
    An optimizer based on SGD with a momentum to smooth
    out the updates.
    """

    def __init__(self, learning_rate: float = 0.5, momentum: float = 0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum

    def calculate(self, w_gradient: np.ndarray, b_gradient: np.ndarray, w_var: np.ndarray, b_var: np.ndarray, 
                  acc_w_grad: np.ndarray, acc_b_grad: np.ndarray, weights: np.ndarray, biases: np.ndarray, 
                  w_momentum: np.ndarray, b_momentum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r'''$$m_{t+1}=\mu m_t + (1-\mu)g_t, \\ \theta_{t+1}=\theta_t-\alpha m_{t+1}$$'''

        w_momentum = (1 - self.momentum) * w_gradient + self.momentum * w_momentum
        b_momentum = (1 - self.momentum) * b_gradient + self.momentum * b_momentum
        return (
            weights - self.learning_rate * w_momentum,
            biases  - self.learning_rate * b_momentum
        )

class AdaGrad(Optimizer):
    """
    <em>Adaptive Gradients</em><br>
    An optimizer which adapts the learning rate for parameters
    based on its historical gradients.
    """

    def __init__(self, learning_rate: float = 0.01, epsilon: float = 1e-9):
        self.learning_rate = learning_rate
        self.epsilon = epsilon

    def calculate(self, w_gradient: np.ndarray, b_gradient: np.ndarray, w_var: np.ndarray, b_var: np.ndarray, 
                  acc_w_grad: np.ndarray, acc_b_grad: np.ndarray, weights: np.ndarray, biases: np.ndarray, 
                  w_momentum: np.ndarray, b_momentum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r'''$$G_{t+1}=G_t+g_t^2, \\ \theta_{t+1}=\theta_t-\frac{\alpha}{\sqrt{G_{t+1}}+\epsilon}g_t$$'''

        acc_w_grad = acc_w_grad + np.square(w_gradient)
        acc_b_grad = acc_b_grad + np.square(b_gradient)
        lr_w = self.learning_rate / np.sqrt(acc_w_grad + self.epsilon)
        lr_b = self.learning_rate / np.sqrt(acc_b_grad + self.epsilon)
        return (
            weights - lr_w * w_gradient,
            biases  - lr_b * b_gradient
        )
    
class RMSprop(Optimizer):
    """
    <em>Root Mean Square Propagation</em><br>
    An adapted version of AdaGrad without rapidly shrinking
    learning rates.
    """

    def __init__(self, learning_rate: float = 0.0005, decay: float = 0.9, epsilon: float = 1e-9):
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon

    def calculate(self, w_gradient: np.ndarray, b_gradient: np.ndarray, w_var: np.ndarray, b_var: np.ndarray, 
                  acc_w_grad: np.ndarray, acc_b_grad: np.ndarray, weights: np.ndarray, biases: np.ndarray, 
                  w_momentum: np.ndarray, b_momentum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r'''$$v_t=\gamma v_{t-1}+(1-\gamma)g^2_t, \\ \theta_{t+1}=\theta_t - \frac{\alpha}{\sqrt{v_t+\epsilon}}g_t$$'''

        acc_w_grad = self.decay * acc_w_grad + (1 - self.decay) * np.square(w_gradient)
        acc_b_grad = self.decay * acc_b_grad + (1 - self.decay) * np.square(b_gradient)
        lr_w = self.learning_rate / np.sqrt(acc_w_grad + self.epsilon)
        lr_b = self.learning_rate / np.sqrt(acc_b_grad + self.epsilon)
        return (
            weights - lr_w * w_gradient,
            biases  - lr_b * b_gradient
        )
    
class Adam(Optimizer):
    """
    <em>Adaptive Moment Estimation</em><br>
    A combined version of the Momentum and RMSprop optimizers.
    """

    def __init__(self, learning_rate: float = 0.001, decay_ma: float = 0.9, decay_sq: float = 0.999, epsilon: float = 1e-9):
        self.learning_rate = learning_rate
        self.decay_ma = decay_ma
        self.decay_sq = decay_sq
        self.epsilon = epsilon

    def calculate(self, w_gradient: np.ndarray, b_gradient: np.ndarray, w_var: np.ndarray, b_var: np.ndarray, 
                  acc_w_grad: np.ndarray, acc_b_grad: np.ndarray, weights: np.ndarray, biases: np.ndarray, 
                  w_momentum: np.ndarray, b_momentum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r'''$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t, \\ v_t=\beta_2v_{t-1}+(1-\beta_2)g^2_t, \\ \hat{m_t}=\frac{m_t}{(1-\beta_1)}, \\ \hat{v_t}=\frac{v_t}{(1-\beta_2)}, \\ \theta_t=\theta_{t-1}-\alpha\frac{\hat{m_t}}{\sqrt{\hat{v_t}}+\epsilon}$$'''

        w_momentum = self.decay_ma * w_momentum + (1 - self.decay_ma) * w_gradient
        b_momentum = self.decay_ma * b_momentum + (1 - self.decay_ma) * b_gradient

        w_var = self.decay_sq * w_var + (1 - self.decay_sq) * np.square(w_gradient)
        b_var = self.decay_sq * b_var + (1 - self.decay_sq) * np.square(b_gradient)
        
        w_momentum_corrected = w_momentum / (1 - self.decay_ma)
        b_momentum_corrected = b_momentum / (1 - self.decay_ma)

        w_var_corrected = w_var / (1 - self.decay_sq)
        b_var_corrected = b_var / (1 - self.decay_sq)

        return (
            weights - w_momentum_corrected / (np.sqrt(w_var_corrected) + self.epsilon) * self.learning_rate,
            biases  - b_momentum_corrected / (np.sqrt(b_var_corrected) + self.epsilon) * self.learning_rate
        )
    
__all__ = ["Optimizer", "SGD", "SGDM", "AdaGrad", "RMSprop", "Adam"]
