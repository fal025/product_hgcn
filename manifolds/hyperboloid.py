import torch
from manifolds.base import Manifold
from torch.autograd import Function
import math

import itertools
from typing import Tuple, Any, Union

torch.autograd.set_detect_anomaly(True)


__all__ = [ "broadcast_shapes" ]
def dot(x,y): return torch.sum(x * y, -1)
def acosh(x):
    return torch.log(x + (x**2-1)**0.5)

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


class Hyperboloid(Manifold):
    """
    Hyperboloid Manifold class.

    We use the following convention: x0^2 + x1^2 + ... + xd^2 < 1 / c

    Note that 1/sqrt(c) is the Poincare ball radius.

    """

    def __init__(self):
        super(Hyperboloid, self).__init__()
        self.name = 'Hyperboloid'
        self.min_norm = 1e-15
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}

    def sqdist(self, p1, p2, c):
        K = c
        d = torch.sqrt(K) * acosh(torch.clamp(-self.inner(p1, c, p1, p2) / K, min =1.+1e-7))
        return d**2

    def egrad2rgrad(self, p, dp, c):
        dp[...,0] *= -1
        dp -= self.inner(p, c, p, dp) / self.inner(p, c, p) * p
        
        return dp
    
    def proj(self, x, c):
        x_ = x.clone()
        x_tail = x_[...,1:].clone().detach()
        current_norms = torch.norm(x_tail,2,-1)
        scale      = (current_norms/1e7).clamp_(min=1.0)
        x_tail /= scale.unsqueeze(-1)
        x_1 = torch.empty(x_.shape)
        x_1[...,1:] = x_tail.clone().requires_grad_(True)
        x_1[...,0] = (torch.sqrt(1 + torch.norm(x_tail,2,-1)**2)).clone().requires_grad_(True)
        xxx = x_1 / torch.sqrt(torch.clamp(-self.inner(x_1, c,x_1,x_1), min=1e-6)).unsqueeze(-1)
        return xxx

    
    def proj_tan(self, u, p, c):
        inner_prod = self.inner(p, c, u,p)
        inner_prod = inner_prod[:,None].repeat(1,u.shape[-1])
        
        return (u + inner_prod * p).clamp(min = 1e-6)


    def proj_tan0(self, u, c):
        return self.proj_tan(u,self.proj(torch.zeros(u.shape).clamp(min = 1e-4),c),c)


    def expmap(self, u, p, c):
        K = c
        u_norm = (self.inner(p, K, u) ** 0.5)
        u_norm = u_norm[:, None].repeat(1,u.shape[-1])
        exp_map = torch.cosh(u_norm / torch.sqrt(K)) * p  \
                + torch.sqrt(K) * torch.sinh(u_norm / torch.sqrt(K)) \
                * (u_norm ** (-1) * u  )
        retr = self.proj(u+p,c)
        cond = u_norm.gt(self.eps[u_norm.dtype])
        return torch.where(cond, exp_map, retr)


    def logmap(self, p1, p2, c):
        K = c
        dist = self.sqdist(p1, p2, c)
        
        u = self.proj_tan(p2,p1,c)
        dist = dist[:,None].repeat(1,p1.shape[-1])
        
        u_norm = self.inner(p1, K, u) ** 0.5
        u_norm = u_norm[:,None].repeat(1,u.shape[-1])

        cond = dist.gt(self.eps[dist.dtype])
        return torch.where(cond, u * dist * (u_norm)**(-1), u)

    def expmap0(self, u, c):
        return self.expmap(u, self.proj(torch.zeros(u.shape).clamp(min = 1e-4),c), c)

    def logmap0(self, p, c):
        return self.logmap(p,self.proj(torch.zeros(p.shape).clamp(min = 1e-4),c),c)

    def mobius_add(self, x, y, c, dim=-1):
        return x + y

    def mobius_matvec(self, m, x, c):
        mx = x @ m.transpose(-1, -2)
        return mx

    def init_weights(self, w, c, irange=1e-5):
        w.data.uniform_(-irange, irange)
        return w

    def inner(self, p, c, u, v=None, keepdim=False, dim=-1):
        if v is None:
            v = u
        inner = torch.sum(u * v, -1, keepdim = keepdim) - 2.*u[...,0]*v[...,0]
        target_shape = broadcast_shapes(p.shape[:-1] + (1,) * keepdim, inner.shape)
        return inner.expand(target_shape)

    def ptransp(self, x, y, u, c):
        ind_distsq = self.sqdist(x, y, 1)
        return u - self.inner(None, c, self.logmap(x,y,1), u) / ind_distsq \
            * (self.logmap(x,y,1) + self.logmap(y,x,1))


    def dot_h(self, x, y):
        return torch.sum(x * y, -1) - 2*x[...,0]*y[...,0]

    def norm_h(self, x):
        assert torch.all(self.dot_h(x,x) >= 0), torch.min(self.dot_h(x,x))
        return torch.sqrt(torch.clamp(self.dot_h(x,x), min=0.0))
    
    def dist_h(self, x, y):
        bad = torch.min(-self.dot_h(x,y) - 1.0)
        if bad <= -1e-4:
            print("bad dist", bad.item())
	    # we're dividing by dist_h somewhere so we can't have it be 0, force dp > 1
        return acosh(torch.clamp(-self.dot_h(x,y), min=(1.0+1e-8)))
