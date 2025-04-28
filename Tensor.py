import numpy as np

class Tensor:

    grad_enabled = True

    def __init__(self,data, requires_grad = False):

        if not isinstance(data, np.ndarray):

            data = np.array(data, dtype = np.float32)

        
        self.data = data
        self.requires_grad = requires_grad and Tensor.grad_enabled
        self.grad = None
        self.isParam = False

        self._backward = lambda: None
        self._prev = set()
        self._op = ""

    def __add__(self, other):

        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data+other.data, requires_grad = self.requires_grad or other.requires_grad)

        def _backward():

            if self.requires_grad and out.grad is not None:

                self.grad = self.grad + out.grad if self.grad is not None else out.grad

            if other.requires_grad and out.grad is not None:

                if out.grad.shape != other.data.shape:
                    out.grad = out.grad.sum(axis=0)

                other.grad = other.grad + out.grad if other.grad is not None else out.grad

        out._backward = _backward
        out._prev = {self, other}
        out._op = "add"

        return out

    def __mul__(self, other):

        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data * other.data, requires_grad = self.requires_grad or other.requires_grad)

        def _backward():

            if self.requires_grad and out.grad is not None:

                self.grad = self.grad + (other.data * out.grad) if self.grad is not None else (other.data* out.grad)

            if other.requires_grad and out.grad is not None:

                grad_other = self.data * out.grad

                if grad_other.shape != other.data.shape:
                    
                    grad_other = grad_other.sum(axis=0)

                other.grad = other.grad + grad_other if other.grad is not None else grad_other

        out._backward = _backward
        out._prev = {self,other}
        out._op = "mul"

        return out

    def __neg__(self):

        out = Tensor(-self.data, requires_grad = self.requires_grad)

        def _backward():
            
            if self.requires_grad and out.grad is not None:

                self.grad= self.grad + (-out.grad) if self.grad is not None else -out.grad

        out._backward = _backward
        out._prev = {self}
        out._op = "neg"

        return out

    def __sub__(self, other):

        other = other if isinstance(other, Tensor) else Tensor(other)

        return self + (-other)

    def __truediv__(self, other):

        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data / other.data, requires_grad = self.requires_grad or other.requires_grad)

        def _backward():

            if self.requires_grad and out.grad is not None:

                self.grad = self.grad + (1/other.data * out.grad) if self.grad is not None else 1/other.data * out.grad

            if other.requires_grad and out.grad is not None:

                grad_other = (-self.data/ other.data **2) * out.grad
                
                if grad_other.shape != other.data.shape:
                    
                    grad_other = grad_other.sum(axis=0)

                other.grad = other.grad + grad_other if other.grad is not None else grad_other

        out._backward = _backward
        out._prev = {self, other}
        out._op = "div"

        return out

    def __pow__(self, exponent):

        out = Tensor(self.data ** exponent, requires_grad = self.requires_grad)

        def _backward():

            if self.requires_grad and out.grad is not None:

                grad = exponent * self.data **(exponent - 1) * out.grad
                self.grad = self.grad + grad if self.grad is not None else grad

        out._backward = _backward
        out._prev = {self}
        out._op = f"pow_{exponent}"

        return out

    def exp(self):


        out = Tensor(np.exp(self.data), requires_grad = self.requires_grad)


        def _backward():
        
            if self.requires_grad and out.grad is not None:

                grad = np.exp(self.data) * out.grad
                self.grad = self.grad + grad if self.grad is not None else grad

        out._backward = _backward
        out._prev = {self}
        out._op = "exp"

        return out

    def log(self):

        out = Tensor(np.log(self.data), requires_grad = self.requires_grad)

        def _backward():

            if self.requires_grad and out.grad is not None:

                grad = (1/self.data) * out.grad
                self.grad = self.grad + grad if self.grad is not None else grad

        out._backward = _backward
        out._prev = {self}
        out._op = "log"

        return out

    def relu(self):

        out = Tensor(np.maximum(0, self.data), requires_grad = self.requires_grad)

        def _backward():

            if self.requires_grad and out.grad is not None:

                grad = (self.data > 0).astype(np.float32) * out.grad
                self.grad = self.grad + grad if self.grad is not None else grad

        out._backward = _backward
        out._prev = {self}
        out._op = "relu"

        return out

    def backward(self):

        if self.grad is None:

            self.grad = np.ones_like(self.data, dtype = np.float32)

        visited = set()
        topo_order = []

        def build_topo(tensor):

            if tensor not in visited:
                
                visited.add(tensor)

                for parent in tensor._prev:

                    build_topo(parent)

                topo_order.append(tensor)

        build_topo(self)

        for tensor in reversed(topo_order):

            tensor._backward()


    def zero_grad(self):

        self.grad = None

    def matmul(self,other):

        other = other if isinstance(other,Tensor) else Tensor(other)

        out = Tensor(self.data @ other.data, requires_grad = self.requires_grad or other.requires_grad)

        def _backward():

            if self.requires_grad and out.grad is not None:

                grad = out.grad @ other.data.T
                self.grad = self.grad + grad if self.grad is not None else grad

            if other.requires_grad:

                grad = self.data.T @ out.grad
                other.grad = other.grad + grad if other.grad is not None else grad

        out.backward = _backward
        out._prev = {self,other}
        out._op = "matmul"

        return out

    def transpose(self):

        transposed_data = self.data.T
        out = Tensor(transposed_data, requires_grad = self.requires_grad)

        def _backward():


            if self.requires_grad and out.grad is not None:


                self.grad = np.zeros_like(self.data)
                self.grad = self.grad + out.grad.T if self.grad is not None else grad

        out._backward = _backward
        out._prev = {self}
        out._op = "transpose"

        return out

    def sum(self, axis = None, keepdims = False):

        data = np.sum(self.data, axis = axis, keepdims = keepdims)
        out = Tensor(data, requires_grad = self.requires_grad)

        def _backward():

            if self.requires_grad and out.grad is not None:

                grad = out.grad
                expanded = grad if axis is None else np.expand_dims(grad, axis = axis) if not keepdims else grad
                self_grad = np.ones_like(self.data) * expanded
                self.grad = self.grad + self_grad if self.grad is not None else self_grad

        out._backward = _backward
        out._prev = {self}
        out._op = "sum"

        return out

    def mean(self, axis = None, keepdims= False):

        data = np.mean(self.data, axis = axis, keepdims = keepdims)
        out = Tensor(data, requires_grad= self.requires_grad)

        def _backward():

            if self.requires_grad and out.grad is not None:

                grad = out.grad
                div = np.prod(self.data.shape) if axis is None else self.data.shape[axis]
                expanded = grad if axis is None else np.expand_dims(grad, axis=axis) if not keepdims else grad
                self_grad = (np.ones_like(self.data) / div) * expanded
                self.grad = self.grad + self_grad if self.grad is not None else self_grad

        out._backward = _backward
        out._prev = {self}
        out._op = "mean"

        return out


    def reshape(self, new_shape):

        reshaped_data = self.data.reshape(new_shape)
        out = Tensor(reshaped_data, requires_grad = self.requires_grad)

        def _backward():

            if self.requires_grad and out.grad is not None:

                grad = out.grad.reshape(self.data.shape)
                self.grad = self.grad + grad if self.grad is not None else grad


        out._backward = _backward
        out._prev = {self}
        out._op = "reshape"

        return out 


    def flatten(self):

        shape = self.data.shape
        batch = shape[0]
        out = self.reshape((batch, -1))
        out._op = "flatten"
        
        return out

    def squeeze(self, axis = None):

        squeezed_data = np.squeeze(self.data, axis = axis)
        out = Tensor(squeezed_data, requires_grad = self.requires_grad)

        def _backward():

            if self.requires_grad and out.grad is not None:

                grad = np.expand_dims(out.grad, axis = axis) if axis is not None else np.reshape(out.grad, self.data.shape)
                self.grad = self.grad + grad if self.grad is not None else grad

        out._backward = _backward
        out._prev = {self}
        out._op = "squeeze"

        return out
    

    def MSELoss(self,target):

        diff = self - target

        out = (diff*diff).mean()

        def _backward():

            if self.requires_grad and out.grad is not None:

                grad = (2*(self.data - target.data))/ np.prod(self.data.shape)
                self.grad = self.grad + grad if self.grad is not None else grad

        out._backward = _backward
        out._prev = {self, target}
        out._op = "mse_loss"

        return out


    def log_softmax(self):

        exps = np.exp(self.data - np.max(self.data, axis=1, keepdims=True))
        softmax = exps / np.sum(exps, axis=1, keepdims=True)
        log_softmax_data = np.log(softmax + 1e-9)
        out = Tensor(log_softmax_data, requires_grad=self.requires_grad)

        def _backward():
            
            if self.requires_grad and out.grad is not None:
                grad = softmax - (np.exp(self.data) / np.sum(np.exp(self.data), axis=1, keepdims=True))
                self.grad = self.grad + grad if self.grad is not None else grad

        out._backward = _backward
        out._prev = {self}
        out._op = "log_softmax"

        return out

    def CrossEntropyLoss(self, target):

        log_probs = self.log_softmax().data
        n = self.data.shape[0]
        loss_val = -np.sum(log_probs[np.arange(n), target.data.astype(int)]) / n
        out = Tensor(loss_val, requires_grad=self.requires_grad)

        def _backward():
            
            if self.requires_grad and out.grad is not None:
                
                exps = np.exp(self.data - np.max(self.data, axis=1, keepdims=True))
                softmax = exps / np.sum(exps, axis=1, keepdims=True)
                grad = softmax.copy()
                grad[np.arange(n), target.data.astype(int)] -= 1
                grad = grad / n
                grad = grad * out.grad
                self.grad = self.grad + grad if self.grad is not None else grad

        out._backward = _backward
        out._prev = {self, target}
        out._op = "cross_entropy"
    
        return out


    def unsqueeze(self, axis):
        
        unsqueezed_data = np.expand_dims(self.data, axis)
        out = Tensor(unsqueezed_data, requires_grad=self.requires_grad)

        def _backward():
            
            if self.requires_grad and out.grad is not None:
                
                grad = np.squeeze(out.grad, axis=axis)
                self.grad = self.grad + grad if self.grad is not None else grad

        out._backward = _backward
        out._prev = {self}
        out._op = "unsqueeze"

        return out
    

    def sigmoid(self):
        sig = 1 / (1 + np.exp(-self.data))
        out = Tensor(sig, requires_grad=self.requires_grad)

        def _backward():
            
            if self.requires_grad and out.grad is not None:
                
                grad = sig * (1 - sig) * out.grad
                self.grad = self.grad + grad if self.grad is not None else grad

        out._backward = _backward
        out._prev = {self}
        out._op = "sigmoid"

        return out
    

    def tanh(self):
        
        tanh_val = np.tanh(self.data)
        out = Tensor(tanh_val, requires_grad=self.requires_grad)

        def _backward():
            
            if self.requires_grad and out.grad is not None:
                grad = (1 - tanh_val ** 2) * out.grad
                self.grad = self.grad + grad if self.grad is not None else grad

        out._backward = _backward
        out._prev = {self}
        out._op = "tanh"

        return out
    

    def clip(self, min_val, max_val):
        
        clipped_data = np.clip(self.data, min_val, max_val)
        out = Tensor(clipped_data, requires_grad=self.requires_grad)

        def _backward():
            
            if self.requires_grad and out.grad is not None:
                mask = (self.data >= min_val) & (self.data <= max_val)
                grad = out.grad * mask.astype(np.float32)
                self.grad = self.grad + grad if self.grad is not None else grad

        out._backward = _backward
        out._prev = {self}
        out._op = "clip"

        return out

    def detach(self):
        
        out = Tensor(self.data.copy(), requires_grad=False)
        out._op = "detach"
        
        return out

    def dropout(self, p = 0.5, training = True):


        if training:

            mask = (np.random.rand(*self.data.shape) > p).astype(np.float32)
            out = Tensor(self.data * mask, requires_grad = self.requires_grad)

            def _backward():

                if self.requires_grad and out.grad is not None:

                    self.grad = self.grad + out.grad * mask if x.grad is not None else out.grad * mask

            out._backward = _backward
            out._prev = {self}
            out._op = "droupout"

            return out

        else: return self

    def max_pool(self, axis = 1):

        max_indices = np.argmax(self.data, axis = axis)
        pooled_data = np.max(self.data, axis = axis)
        out = Tensor(pooled_data, requires_grad = self.requires_grad)

        def _backward():

            if self.requires_grad and out.grad is not None:

                grad = np.zeros_like(self.data)
                idx = np.indices(max_indices.shape)
                idx = list(idx)
                idx.insert(axis, max_indices)

                grad[tuple(idx)] = out.grad
                self.grad = self.grad + grad if self.grad is not None else grad

        out._backward = _backward
        out._prev = {self}
        out._op = "max_pool"

        return out


    def __repr__(self):

        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"
    

