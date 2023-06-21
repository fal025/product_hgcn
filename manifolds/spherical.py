import itertools
import torch

from manifolds.base import Manifold

def broadcast_shapes(*shapes):
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
    def __init__(self, ):
        super(Spherical, self).__init__()
        self.name = 'Spherical'
        self.min_norm = 1e-15
        self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}

    def dist(self, x, y, c):
        return torch.arccos(torch.inner(x, y))

    def sqdist(self, x, y, c):
        return self.dist(x, y, c) ** 2

    def egrad2rgrad(self, p, dp, c):
        return self.proju(p, dp)

    def proj(self, x, c):
        return x / torch.norm(x).clamp_min(self.eps[x.dtype])
    
    def proju(self, x, u, c):
        u = u - (x * u).sum(dim=-1, keepdim=True) * x
        return u

    def proj_tan(self, u, p, c):
        return u

    def proj_tan0(self, u, c):
        return u

    def expmap(self, u, p, c):
        u_norm = u.norm(dim=-1, keepdim=True).clamp_min(self.eps[u.dtype])
        exp = p * torch.cos(u_norm) + u * torch.sin(u_norm) / u_norm
        retr = self.proj(p + u, c)
        cond = u_norm > self.eps[u_norm.dtype]
        return torch.where(cond, exp, retr)

    def logmap(self, p, q, c):
        res = torch.empty_like(p)
        for x, y in zip(p, q):
            dist = self.dist(x, y, c)
            num = self.proju(x, y - x, c)
            cond = dist.gt(self.eps[x.dtype])
            val = dist *  num / torch.norm(num, dim=-1, keepdim=True).clamp_min(self.eps[x.dtype])
            filt = torch.where(cond, val, num)
            torch.cat((res, filt.unsqueeze(0)), dim=0)
        return res

    def expmap0(self, u, c):
        orig = torch.zeros(u.size())
        orig[-1] = 1
        return self.expmap(u, orig, c)

    def logmap0(self, p, c):
        orig = torch.zeros(p.size())
        orig[-1] = 1
        return self.logmap(p, orig, c)

    def mobius_add(self, x, y, c, dim=-1):
        return x + y

    def mobius_matvec(self, m, x, c):
        mx = x @ m.transpose(-1, -2)
        return mx

    def init_weights(self, w, c, irange=1e-5):
        w.data.uniform_(-irange, irange)
        return w

    def inner(self, x, c, u, v=None, keepdim=False, dim=-1):
        if v is None:
            v = u
        inner = (u * v).sum(-1, keepdim=keepdim)
        target_shape = broadcast_shapes(x.shape[:-1] + (1,) * keepdim, inner.shape)
        return inner.expand(target_shape)

    def ptransp(self, x, y, u, c):
        denom = self.sqdist(x, y, c)
        num = self.logmap(x, y, c) @ u
        term = self.logmap(x, y, c) + self.logmap(y, x, c)
        return u - (num / denom) * term
