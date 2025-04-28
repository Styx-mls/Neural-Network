import pickle 
import numpy as np
from Tensor import Tensor


class Module:

    def __init__(self):

        self.training = True

    def parameters(self):

        params = []
        
        for val in self.__dict__.values():
            
            if isinstance(val, Module):
                
                params += val.parameters()
                
            elif isinstance(val, (list, tuple)):
                
                for item in val:
                    
                    if isinstance(item, Module):
                        
                        params += item.parameters()
                        
            elif isinstance(val, Tensor) and val.isParam:
                
                params.append(val)
                
        return params

    def zero_grad(self):

        for p in self.parameters():

            p.grad = None

    def train(self):

        self.training = True

    def eval(self):

        self.training = False

    def __call__(self, *args, **kwargs):

        return self.forward(*args, **kwargs)

    def save(self, filepath):

        with open(filepath, "wb") as f:

            pickle.dump([param.data for param in self.parameters()], f)

    def load(self, filepath):

        with open(filepath, "rb") as f:

            state = pickle.load(f)

            for param, data in zip(self.parameters(), state):

                param.data[:] = data
    
    def forward(self, *args, **kwargs):

        raise NotImplementedError("Must implement forward method")


class Linear(Module):

    def __init__(self, in_features, out_features):

        super().__init__()
        self.weight = Tensor(np.random.randn(out_features, in_features) * np.sqrt(2./ in_features), requires_grad = True)
        self.weight.isParam = True
        self.bias = Tensor(np.zeros(out_features), requires_grad = True)
        self.bias.isParam = True

    def forward(self,x):

        og_shape = x.data.shape
        feature_dim = x.data.shape[-1]
        x_flat = x.reshape((-1, feature_dim))

        out = x_flat.matmul(self.weight.transpose()) + self.bias

        new_shape = og_shape[:-1] + (self.weight.data.shape[0],)
        out = out.reshape(new_shape)

        return out
        

class ReLU(Module):

    def forward(self, x):

        return x.relu()

class Tanh(Module):

    def forward(self,x):

        return x.tanh()

class Sigmoid(Module):

    def forward(self, x):

        return x.sigmoid()

class Sequential(Module):

    def __init__(self, *layers):

        super().__init__()
        self.layers = layers

    def forward(self, x):

        for layer in self.layers:

            x = layer(x)

        return x

    def parameters(self):

        params = []

        for layer in self.layers:
            
            if isinstance(layer, Module):
                
                params = params + layer.parameters()

        return params

class Dropout(Module):

    def __init__(self, p = 0.5):

        super().__init__()
        self.p = p

    def forward(self, x):

        return x.dropout(self.p, training = self.training)


class MSELoss(Module):

    def forward(self, x, target):

        return x.MSELoss(target)

class CrossEntropyLoss(Module):

    def forward(self, x, target):

        return x.CrossEntropyLoss(target)

class Flatten(Module):

    def forward(self, x):

        return x.flatten()

class Residual(Module):

    def __init__(self, fn):

        super().__init__()
        self.fn = fn

    def forward(self, x):

        return x + self.fn(x)

class no_grad:

    def __enter__(self):

        Tensor.grad_enabled = False

    def __exit__(self, exc_type, exc_val, exc_tb):

        Tensor.grad_enabled = True


