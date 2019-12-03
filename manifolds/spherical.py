import torch
from manifolds.base import Manifold
from torch.autograd import Function
from utils.math_utils import artanh, tanh
import itertools
from typing import Tuple, Any, Union
__all__ = [ "broadcast_shapes" ]
def dot(x,y): return torch.sum(x * y, -1)
def broadcast_shapes(*shapes: Tuple[int]) -> Tuple[int]:
    """Apply numpy broadcasting rules to shapes."""
    result = []
    for dims in itertools.zip_longest(*map(reversed, shapes), fillvalue=1):
        dim: int = 1
        for d in dims:
            if dim != 1 and d != 1 and d != dim:
                raise ValueError("Shapes can't be broadcasted")
            elif d > dim:
                dim = d
        result.append(dim)
    return tuple(reversed(result))

class Spherical(Manifold):
    """
    Spherical Manifold class.
    """



    def __init__(self):
        super(Spherical, self).__init__()
        self.name = 'Spherical'
        self.eps = {torch.float32: 1e-4, torch.float64: 1e-6}
        #print("-----------")

    def normalize(self, p ):
        dim = p.size(-1)
        p.view(-1, dim).renorm_(2, 0, 1.)
        return p

    # this function compute the square distance for p1, p2
    def angle (self, p1, p2, c=1):
        
        #print(p1)
        inner_prod = (p1 * p2).sum(dim=-1, keepdim=True)
        return torch.acos(inner_prod.clamp(min = -.9999, max = .9999) / (c**2))
        
        
    def sqdist(self, p1, p2,c=1):
        
        return self.dist(p1,p2) ** 2

    def egrad2rgrad(self, p, dp, c=1):
        
        #######
        
        return proj_tan(dp, p, c=1)

    def proj(self, p, c):
        #print("project")
        #print("proj")
        #print(p.sum())
        
        return p / p.norm(2, dim=-1, keepdim=True).clamp(min = 1e-6)

    def proj_tan(self, u, p, c=1):

        u = u - (p * u).sum(dim=-1, keepdim=True) * p
        #print("proj_tan")
        #print(u.sum())
        #print("-------")
        return u

    def proj_tan0(self, u, c=1):
        
        unit = self.proj(torch.ones(u.shape),c)

        return self.proj_tan(u,unit,c)

    def expmap(self, u, p, c=1):

        
        #print("in exp")
        #print((p).sum())
        #print(u.sum())
        #print("----------")
        #print(u)
        #print(u.shape)
        #print(p.shape)
        norm_u = u.norm(dim=-1, keepdim=True).clamp(min = 1e-7)
        exp = p * torch.cos(norm_u) + u * torch.sin(norm_u) / norm_u
        retr = self.proj(p + u,c)
        #print("in exp after proj")
        #print(torch.isnan(exp).sum())
        cond = norm_u > self.eps[norm_u.dtype]
        
        #print("expmap")
        #print(exp.shape)



        return torch.where(cond, exp, retr)

    def logmap(self, p1, p2, c=1):
        u = self.proj_tan(p1, p2 - p1,c)
        #print("logmap")
        dist = self.dist(p1, p2, keepdim = True)
        #print(dist)
        #print("-------")
        # If the two points are "far apart", correct the norm.
        cond = dist.gt(self.eps[dist.dtype])

        #print(cond)
        #print("------")
        #print(dist.shape)
        #print(u.shape)
        #print(u*dist)
        #print(torch.isnan(u * dist / u.norm(2,dim=-1, keepdim=True)).sum())
        return torch.where(cond, u * dist / u.norm(2,dim=-1, keepdim=True).clamp(min = 1e-8), u)

    def expmap0(self, u, c=1):
        
        return self.expmap(u, self.proj(torch.ones(u.shape),c), c)
    

    def logmap0(self, p, c=1):


        return self.logmap(p, self.proj(torch.ones(p.shape),c),c)


    def init_weights(self, w, c=1, irange=1e-5):
        w.data.uniform_(-irange, irange)
        return w

    def inner(self,p, c, u, v=None, keepdim=False):
        if v is None:
            v = u

        #print(u)
        #print(v)
        #print( torch.cos(self.angle(u,v, c)))
        #return torch.cos(self.angle(u,v, c))


        if v is None:
            v = u

        #print(u)
        #print("----------")
        #print(v)
        #print("++++++++++")
        inner = (u * v).sum(-1, keepdim=keepdim)
        #print(p.shape[:-1])
        target_shape = broadcast_shapes(p.shape[:-1] + (1,) * keepdim, inner.shape)
        #print(target_shape)
        #print(inner.expand(target_shape))
        return inner.expand(target_shape)

    def ptransp(self, x, y, v, c=1):
    
        return self.proj_tan(y,v)

    def dist(self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False) -> torch.Tensor:
        #print(x)
        #print(y)
        #uu = self.proj(x,1.)
        #print(uu.shape)
        #vv = self.proj(y,1.)
        #dot_prod = dot(uu, vv)
        #print(dot_prod.shape)
        #return torch.acos(torch.clamp(dot_prod, -1+self.eps[dot_prod.dtype], 1-self.eps[dot_prod.dtype]))
        
        
        #print(y)
        #print("_________")
        inner = self.inner(x,1, x, y, keepdim=keepdim).clamp(-.999999, .999999)
        #print(inner.shape)
        #print(torch.isnan(inner).sum())
        return torch.acos(inner)
    
    def mobius_add(self, x, y, c=1, dim=-1):
        return x + y

    def mobius_matvec(self, m, x, c=1):
        mx = x @ m.transpose(-1, -2)
        return mx
