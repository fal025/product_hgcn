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

    #def egrad2rgrad(self, p, dp, c):
        lambda_p = self._lambda_x(p, c)
        dp /= lambda_p.pow(2)
        return dp

    #def proj(self, x, c):
        norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), self.min_norm)
        maxnorm = (1 - self.eps[x.dtype]) / (c ** 0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    def proj_tan(self, u, p, c):
        return u

    def proj_tan0(self, u, c):
        return u

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

    #def expmap0(self, u, c):
        sqrt_c = c ** 0.5
        u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), self.min_norm)
        gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
        return gamma_1

    #def logmap0(self, p, c):
        sqrt_c = c ** 0.5
        p_norm = p.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        scale = 1. / sqrt_c * artanh(sqrt_c * p_norm) / p_norm
        return scale * p

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

    #def ptransp(self, x, y, u, c):
        lambda_x = self._lambda_x(x, c)
        lambda_y = self._lambda_x(y, c)
        return self._gyration(y, -x, u, c) * lambda_x / lambda_y
