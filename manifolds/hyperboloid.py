import torch
from manifolds.base import Manifold
from torch.autograd import Function
from utils.math_utils import artanh, tanh, cosh, sinh, arcosh

class Hyperboloid(Manifold):
    """
    Hyperboloid Manifold class.

    We use the following convention: x0^2 + x1^2 + ... + xd^2 < 1 / c

    Note that 1/sqrt(c) is the Poincare ball radius.

    """

    def __init__(self, ):
        super(Hyperboloid, self).__init__()
        self.name = 'Hyperboloid'
        self.min_norm = 1e-15
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}

    def sqdist(self, p1, p2, c):
        K = c
        d = torch.sqrt(K) * arcosh(-self.inner(None, c, p1, p2) / K)
        return d**2

    def egrad2rgrad(self, p, dp, c):
        return self.proj_tan(dp, p, c)

    def proj(self, x, c):
        replace = torch.sqrt(1+torch.norm(x[1:],p = 2)**2)
        x[0] = replace
        return x

    def proj_tan(self, u, p, c):
        return p + self.inner(None, c, u, p) * u

    def proj_tan0(self, u, c):
        return self.proj_tan(u,0,c)

    def expmap(self, u, p, c):
        K = c
        exp_map = cosh(self.inner(p, K, u) / torch.sqrt(K)) * p  \
            + torch.sqrt(K) * sinh(self.inner(p, K, u) / torch.sqrt(K)) \
                * (u / self.inner(p, K, u))
        return exp_map

    def logmap(self, p1, p2, c):
        K = c
        log_map = self.sqdist(p1, p2, c) * (p2 + 1/K * self.inner(None, c, p1, p2)*p1) \
            / self.inner(None, c, (p2 + 1/K * self.inner(None, c, p1, p2)*p1))
        return log_map

    def expmap0(self, u, c):
        return self.expmap(u, 0, c)

    def logmap0(self, p, c):
        return self.logmap(p,0,c)

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
        return (u * v).sum(dim=dim, keepdim=keepdim) - 2 * u[0] * v[0]

    def ptransp(self, x, y, u, c):
        ind_distsq = self.sqdist(x, y, 1)
        return u - self.inner(None, c, self.logmap(x,y,1), u) / ind_distsq \
            * (self.logmap(x,y,1) + self.logmap(y,x,1))


