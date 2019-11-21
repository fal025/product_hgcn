import torch
from manifolds.base import Manifold
from torch.autograd import Function
from utils.math_utils import artanh, tanh
from manifolds.base import Manifold



class Spherical(Manifold):
    """
    Spherical Manifold class.
    """

    def __init__(self):
        super(Euclidean, self).__init__()
        self.name = 'Spherical'
        self.min_norm = 1e-15
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}

    # this function compute the square distance for p1, p2
    def angle (self, p1, p2, c):

        inner_prod = (p1 * p2).sum(dim=-1, keepdim=keepdim)


        return torch.acos(inner_prod / (c**2))
        
        
    def sqdist(self, p1, p2, c):
        
        return self.inner(p1,p2,c)

    def egrad2rgrad(self, p, dp, c):
        
        #######
        
        return proj_tan(dp, p, c)

    def proj(self, p, c):

        
        return p / p.norm(dim=-1, keepdim=True()

    def proj_tan(self, u, p, c):

        u = u - (p * u).sum(dim=-1, keepdim=True) * p
        
        return u

    def proj_tan0(self, u, c):
    
        return u

    def expmap(self, u, p, c):

        norm_u = u.norm(dim=-1, keepdim=True)
        exp = x * torch.cos(norm_u) + u * torch.sin(norm_u) / norm_u
        retr = self.proj(x + u)
        cond = norm_u > self.eps[norm_u.dtype]

        return torch.where(cond, exp, retr)

    def logmap(self, p1, p2, c):
        
        
    
        return p2 - p1

    def expmap0(self, u, c):
        return u

    def logmap0(self, p, c):
        return p

    def mobius_add(self, x, y, c, dim=-1):
        
        return x + y

    def mobius_matvec(self, m, x, c):
        mx = x @ m.transpose(-1, -2)
        return mx

    def init_weights(self, w, c, irange=1e-5):
        w.data.uniform_(-irange, irange)
        return w

    def inner(self, c, u, v=None, keepdim=False):
        if v is None:
            v = u

        return  c**2 * torch.cos(self.angle(u,v, c))

    def ptransp(self, x, y, v, c):
        return v
