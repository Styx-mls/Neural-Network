import numpy as np

class SGD:

    def __init__(self, parameters, lr = 0.01):

        self.parameters = parameters
        self.lr = lr

    def step(self):

        for p in self.parameters:

            if p.grad is not None:

                p.data = p.data - (self.lr * p.grad)

    def zero_grad(self):

        for p in self.parameters:

            p.grad = None

class Adam:

    def __init__(self, parameters, lr = 0.01, betas = (0.9, 0.999), eps = 1e-8):

        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.epsilon = eps
        self.t = 0
        self.m = {}
        self.v = {}

        for i, p in enumerate(self.parameters):

            self.m[i] = np.zeros_like(p.data)
            self.v[i] = np.zeros_like(p.data)


    def step(self):

        self.t = self.t + 1

        for i, p in enumerate(self.parameters):

            if p.grad is None:

                continue

            self.m[i] = self.beta1 * self.m[i] + (1-self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1-self.beta2) * (p.grad ** 2)

            m_unit = self.m[i] / (1-self.beta1 ** self.t)
            v_unit = self.v[i] / (1-self.beta2 ** self.t)

            p.data = p.data - self.lr * m_unit / (np.sqrt(v_unit) + self.epsilon)

    def zero_grad(self):

        for p in self.parameters:

            p.grad = None


