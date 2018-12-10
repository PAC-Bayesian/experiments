import numpy as np
import inspect
import scipy as sp


### 1d functions
bounds_1d = ((0, 1),) # set x range

def f_1d_zero(x):
    return x * 0

def f_1d_single_flat(x):
    return -np.sin(0.5 * (x + 2.65)) + 0.5

def f_1d_single_steep(x):
    return -np.sin(3 * x)

def f_1d_double(x):
    return -np.sin(10.5 * (x - 0.05))

def f_1d_triple(x):
    return -np.sin(18 * x) 

def f_1d_forrester(x):
    return (6 * x - 2)**2 * np.sin(12 * x - 4) / 15

def f_1d_poly_4(x):
    return 64 * x * (x - 1/4) * (x - 3/4) * (x - 1) 

def f_1d_poly_2(x):
    return 8 * (x - 0.5)**2 - 1

def f_1d_linear(x):
    return 2 * x - 1

def f_1d_abs(x):
    return 4 * np.abs(x - 0.5) - 1


### 2d functions
bounds_2d = ((0, 1), (0, 1)) # set x range and y range

def f_2d_single(x, y):
    return -np.sin(3 * x) * np.sin(3 * y)

def f_2d_double(x, y):
    return -1.25 * np.sin(6 * x) * np.sin(6 * y)


### Additive functions
class AdditiveFun(object):
    """Construct an additive function."""

    def __init__(self, funs):
        """
        : param: list or tuple
                func[i]: function
                a sequence of functions
        """
        assert isinstance(funs, (list, tuple)) and len(funs) >= 1, "cannot make additive"
        self.funs = funs
        self.num_comp = len(funs)
        self.dims = ()
        self._collect_dims()
    
    def _collect_dims(self):
        """Combine dimensions of each component function."""
        for f in self.funs:
            sig = inspect.signature(f)
            self.dims += (len(sig.parameters),)
    
    def add_f(self):
        """Return the additive function."""
        def f_true(*args):
            result = 0
            start = 0
            for i in range(self.num_comp):
                end = start + self.dims[i]
                result += self.funs[i](*args[start:end])
                start = end
            return result
        return f_true

    def get_input_dim(self):
        """Sum all dimensions together."""
        return sum(self.dims)

    def __repr__(self):
        return "additive func of {} components \
        with dims: {}".format(self.num_comp, self.dims)