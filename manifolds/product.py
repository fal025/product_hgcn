import torch
import itertools

from typing import Tuple, Any, Union
from manifolds.base import Manifold

def make_tuple(obj: Union[Tuple, Any]) -> Tuple:
    if not isinstance(obj, tuple):
        return (obj,)
    else:
        return obj

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

def _calculate_target_batch_dim(*dims: int):
    return max(dims) - 1

class Product(Manifold):
    """
    A product manifold, made up of 
    Spherical, Hyperbolic, and Euclidean 
    components.
    """
    def __init__(self, manifolds, total_dim=105):
        super().__init__()
        self.man_count = manifolds
        self.manifolds = [x[0] for x in manifolds]
        self.name = "Product"

        self.total_dim = total_dim
        self.n_manifolds = sum([x[1] for x in manifolds])

        self.indices = self.calc_indices(manifolds, total_dim)

    def calc_indices(self, manifolds, total_dim, first_iter=True):
        self.indices = []
        if first_iter is False:
            indiv_dim = total_dim // len(self.manifolds)
            total = 0
            print(indiv_dim)
            for i, man in enumerate(self.manifolds):
                self.indices.append((i * indiv_dim, (i + 1) * indiv_dim))

        else:
            print(f"n manifolds: {self.n_manifolds}")
            indiv_dim = total_dim // self.n_manifolds
            total = 0
            for j, (man, count) in enumerate(manifolds):
                self.indices.append([])
                for i in range(count):
                    if j == 2 and i == count - 1:
                        self.indices[j].append((total, total_dim))
                    else:
                        self.indices[j].append((total, total + indiv_dim))
                    total += indiv_dim

    def split_input(self, *args):
        split = []
        print(f"indices: {self.indices}")
        for man in self.indices:
            man_split = []
            for s in man:
                split_arg = tuple([arg[:, s[0]:s[1]] for arg in args])
                man_split.append(split_arg)
            split.append(man_split)
        return split


    def sqdist(self, p1, p2, c):
        """Squared distance between pairs of points."""
        self.calc_indices(self.man_count, p1.size(1), first_iter=True)
        splits = self.split_input(p1, p2)
        res = []
        for i, man in enumerate(self.manifolds):
            man_split = splits[i]
            for s in man_split:
                res.append(man.sqdist(*s, c))
        res[0] = res[0].squeeze()
        return torch.cat(res[:-1], dim=0)

    def egrad2rgrad(self, p, dp, c):
        """Converts Euclidean Gradient to Riemannian Gradients."""
        self.calc_indices(self.man_count, p.size(1), first_iter=True)
        splits = self.split_input(p, dp)
        res = []
        for i, man in enumerate(self.manifolds):
            man_split = splits[i]
            for s in man_split:
                res.append(man.proj(*s, c))
        return torch.cat(res, dim=1)

    def proj(self, p, c):
        """Projects point p on the manifold."""
        self.calc_indices(self.man_count, p.size(1), first_iter=True)
        splits = self.split_input(p)
        res = []
        for i, man in enumerate(self.manifolds):
            man_split = splits[i]
            for s in man_split:
                res.append(man.proj(*s, c))
        return torch.cat(res, dim=1)

    def proj_tan(self, u, p, c):
        """Projects u on the tangent space of p."""
        self.calc_indices(self.man_count, u.size(1), first_iter=True)
        splits = self.split_input(u, p)
        res = []
        for i, man in enumerate(self.manifolds):
            man_split = splits[i]
            for s in man_split:
                res.append(man.proj_tan(*s, c))
        return torch.cat(res, dim=1)

    def proj_tan0(self, u, c):
        """Projects u on the tangent space of the origin."""
        self.calc_indices(self.man_count, u.size(1), first_iter=True)
        splits = self.split_input(u)
        res = []
        for i, man in enumerate(self.manifolds):
            man_split = splits[i]
            for s in man_split:
                res.append(man.proj_tan0(*s, c))
        return torch.cat(res, dim=1)

    def expmap(self, u, p, c):
        """Exponential map of u at point p."""
        self.calc_indices(self.man_count, u.size(1), first_iter=True)
        splits = self.split_input(u, p)
        res = []
        for i, man in enumerate(self.manifolds):
            man_split = splits[i]
            for s in man_split:
                res.append(man.expmap(*s, c))
        return torch.cat(res, dim=1)

    def logmap(self, p1, p2, c):
        self.calc_indices(self.man_count, p1.size(1), first_iter=True)
        splits = self.split_input(p1, p2)
        res = []
        for i, man in enumerate(self.manifolds):
            man_split = splits[i]
            for s in man_split:
                res.append(man.logmap(*s, c))
        return torch.cat(res, dim=1)

    def expmap0(self, u, c):
        """Exponential map of u at point p."""
        self.calc_indices(self.man_count, u.size(1), first_iter=True)
        splits = self.split_input(u)
        res = []
        for i, man in enumerate(self.manifolds):
            man_split = splits[i]
            for s in man_split:
                res.append(man.expmap0(*s, c))
        return torch.cat(res, dim=1)

    def logmap0(self, p, c):
        self.calc_indices(self.man_count, p.size(1), first_iter=True)
        splits = self.split_input(p)
        res = []
        for i, man in enumerate(self.manifolds):
            man_split = splits[i]
            for s in man_split:
                res.append(man.logmap0(*s, c))
        return torch.cat(res, dim=1)

    def mobius_add(self, x, y, c, dim=-1):
        self.calc_indices(self.man_count, x.size(1))
        splits = self.split_input(x, y)
        res = []
        for i, man in enumerate(self.manifolds):
            man_split = splits[i]
            for s in man_split:
                res.append(man.mobius_add(*s, c))
        return torch.cat(res, dim=1)

    def mobius_matvec(self, m, x, c):
        self.calc_indices(self.man_count, m.size(1))
        splits = self.split_input(m, x)
        res = []
        for i, man in enumerate(self.manifolds):
            man_split = splits[i]
            for s in man_split:
                print(f"len: {len(s)}")
                print(f"hee: {s}")
                for x in s:
                    print(x.size())
                res.append(man.mobius_matvec(*s, c))
        return torch.cat(res, dim=1)

    def init_weights(self, w, c, irange=1e-5):
        """Initializes random weigths on the manifold."""
        self.calc_indices(self.man_count, w.size(1))
        splits = self.split_input(w)
        res = []
        for i, man in enumerate(self.manifolds):
            man_split = splits[i]
            for s in man_split:
                res.append(man.init_weights(*s, c))
        return torch.cat(res, dim=1)

    def inner(self, p, c, u, v=None, keepdim=True):
        """Inner product for tangent vectors at point x."""
        self.calc_indices(self.man_count, p.size(1))
        splits = self.split_input(p, u)
        res = []
        for i, man in enumerate(self.manifolds):
            man_split = splits[i]
            for s in man_split:
                res.append(man.inner(*s, c))
        return torch.cat(res, dim=1)

    def ptransp(self, x, y, u, c):
        self.calc_indices(self.man_count, x.size(1))
        splits = self.split_input(x, y, u)
        res = []
        for i, man in enumerate(self.manifolds):
            man_split = splits[i]
            for s in man_split:
                res.append(man.ptransp(*s, c))
        return torch.cat(res, dim=1)
    
    def take_submanifold_value(self, x, i, reshape=True, is_matvec=False):
        x = x.clone()
        slc_length = int(x.size(-1) / self.num_spaces)
        if is_matvec:
            slc_length_col = int(x.size(-2) / self.num_spaces)
        slc = self.slices[i]
        start = slc.start * slc_length
        length =  (slc.stop - slc.start) * slc_length
        if x.size(-1) - (start + slc_length) < slc_length:
            length = x.size(-1) - start

        if is_matvec:
            start_col = slc.start * slc_length_col
            length_col =  (slc.stop - slc.start) * slc_length_col
            if x.shape(-2) - (start_col + slc_length_col)< slc_length_col:
                length_col = x.shape(-2) - start_col

        if not is_matvec:
            part = x.narrow(-1, start, length)
        else:
            part = torch.zeros((length_col,length)) + x[start_col:start_col+length_col, start:start+length]
        return part

    def normalize(self, p):
        target_batch_dim = _calculate_target_batch_dim(p.dim())
        new_p = []
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(p, i)
            if manifold.name == "Euclidean":
                point = manifold.normalize(point)
            
            point = point.reshape(
                (*point.shape[:target_batch_dim], -1)
            )
            new_p.append(point)
        res = torch.cat(new_p, -1)
        return res
