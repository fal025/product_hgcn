import torch
import itertools

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
    def __init__(self):
        super().__init__()
        self.name = 'Spherical'
        self.min_norm = 1e-15
        self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}

    def inner_product(self, u, v):
        # return torch.tensordot(u, v, dims=2)
        # print(u.size(), v.size())
        if u.size(1) != v.size(1):
            v = v.T
        return torch.diag(torch.inner(u.T, v.T))

    def dist(self, x, y, c):
        # print(f"x: {x.size()}")
        # print(f"y: {y.size()}")
        # print(f"inner: {self.inner_product(x,y)}")
        return torch.arccos(self.inner_product(x, y).clamp(0.0, torch.pi))

    def sqdist(self, x, y, c):
        dist = self.dist(x, y, c) ** 2
        # print(f"dist: {dist}")
        return dist

    def egrad2rgrad(self, p, dp, c):
        return self.proju(p, dp)

    def proj(self, x, c):
        projected = x / torch.norm(x).clamp_min(self.eps[x.dtype])
        # print(f"proj: {projected}")
        return projected

    def proju(self, x, u, c):
        # print(f"x size: {x.size()}")
        return u - self.inner_product(x, u) * x

    def expmap(self, v, p, c):
        p = self.proj(p, 1.0)
        # print(f"p norm: {torch.linalg.norm(p)}")
        v = torch.nan_to_num(v)
        v_norm = abs(c) * v.norm(dim=-1, keepdim=True).clamp(self.eps[v.dtype], 1000)
        # print(f"cos(v_norm): {torch.cos(v_norm)}")
        exp = p * torch.cos(v_norm) + v * c * torch.sinc(v_norm / torch.pi)
        # print(f"exp: {exp}")
        return exp

    def logmap(self, x, y, c):
        x = self.proj(x, 1.0)
        y = self.proj(y, 1.0)
        # print(f"y: {y}")
        v = self.proju(x, y - x, c)
        # print(f"v: {v}")
        # print(f"x norm: {torch.linalg.norm(x)}")
        # print(f"y norm: {torch.linalg.norm(y)}")
        dist = self.dist(x, y, c)
        # print(f"dist: {dist}")
        eps = self.eps[x.dtype]
        scale = (dist + eps) / (torch.linalg.norm(v) + eps)
        # print(f"scale: {scale}")
        # print(f"scale * v: {scale * v}")
        return scale * v

    def expmap0(self, u, c):
        orig = torch.zeros(u.size())
        orig[-1] = 1
        return self.expmap(u, orig, c)

    def logmap0(self, p, c):
        orig = torch.zeros(p.size(), dtype=p.dtype)
        orig[-1] = 1
        return self.logmap(orig, p, c)

    def mobius_add(self, x, y, c, dim=-1):
        u = self.logmap0(y, c)
        v = self.ptransp0(x, u, c)
        return self.expmap(v, x, c)

    def mobius_matvec(self, m, x, c):
        u = self.logmap0(x, c)
        mu = u @ m.T
        return self.expmap0(mu, c)

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
        m = x + y
        m_norm = self.inner_product(m, m)
        factor = 2 * self.inner_product(u, y) / m_norm
        transported = u - m * factor
        # print(f"transported: {transported}")
        return transported

    def proj_tan(self, u, x, c):
        x = x.clone()
        x[:, -1] = torch.diag(x[:, :-1] @ u[:, :-1].T) / u[:, -1]
        # print(f"tan: {x}")
        return x

    def proj_tan0(self, x, c):
        orig = torch.zeros(x.size(), dtype=x.dtype)
        orig[-1] = 1
        return self.proj_tan(orig, x, c)


    # x2[2] = -torch.inner(x2[:2], s[:2]) / s[2]