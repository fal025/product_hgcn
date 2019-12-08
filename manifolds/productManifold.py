from __future__ import division
import torch
from torch.nn import Parameter
import numpy as np
from manifolds.base import Manifold
import itertools
from torch.autograd import Function
from typing import Tuple, Any, Union
"""
    this class is referred from the package geoopt's product manifold with necessary 
    modification
"""
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

def size2shape(*size: Union[Tuple[int], int]) -> Tuple[int]:
    return make_tuple(strip_tuple(size))

def _calculate_target_batch_dim(*dims: int):
    return max(dims) - 1

class productManifold(Manifold):
    """
    Abstract class to define operations on a manifold.
    """

    def __init__(self, manifolds, total_dim):
        super(productManifold,self).__init__()
        self.manifolds = [x[0] for x in manifolds]
        self.name = "productManifold"

        self.num_man = len(manifolds)

        self.slices = []
        self.total_ratio = 0
        #dtype = None
        pos0 = 0
        for i, manifold in enumerate(manifolds):

            #print(manifold[1])
            pos1 = pos0 + manifold[1]
            self.total_ratio += manifold[1]
            self.slices.append(slice(pos0, pos1))
            pos0 = pos1
        #print(self.total_ratio)

    def sqdist(self, p1, p2, c):
        """Squared distance between pairs of points."""
        target_batch_dim = _calculate_target_batch_dim(p1.dim(), p2.dim())
        mini_dists2 = []
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(p1, i)
            point1 = self.take_submanifold_value(p2, i)
            mini_dist2 = manifold.sqdist(point, point1, c)
            mini_dist2 = mini_dist2.reshape(
                (*mini_dist2.shape[:target_batch_dim], -1)
            ).sum(-1)
            mini_dists2.append(mini_dist2)
        result = sum(mini_dists2).clamp(max = 75)
        return result

    def egrad2rgrad(self, p, dp, c):
        """Converts Euclidean Gradient to Riemannian Gradients."""
        target_batch_dim = _calculate_target_batch_dim(p.dim(), dp.dim())
        transformed_tensors = []
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(p, i)
            grad = self.take_submanifold_value(dp, i)
            transformed = manifold.egrad2rgrad(point, grad)
            transformed = transformed.reshape(
                (*transformed.shape[:target_batch_dim], -1)
            )
            transformed_tensors.append(transformed)
        res = torch.cat(transformed_tensors, -1)
        #if not res.requires_grad:
        #    raise ValueError



        return res

    def proj(self, p, c):
        """Projects point p on the manifold."""
        projected = []

        #print(self.manifolds)
        for i, manifold in enumerate(self.manifolds):
            #print("proj")
            #print(p.shape)
            point = self.take_submanifold_value(p, i)

            #print(point == p)
            #print(point)
            proj = manifold.proj(point,c)
            #proj.requires_grad = True
            
            proj = proj.reshape(*p.shape[: len(p.shape) - 1], -1)
            projected.append(proj)
            #print(proj)
        #print(projected)

        res =  torch.cat(projected, -1)
        #print(res.shape)
        #if not res.requires_grad:
        #    raise ValueError


        return res

    def proj_tan(self, u, p, c):
        """Projects u on the tangent space of p."""
        target_batch_dim = _calculate_target_batch_dim(u.dim(), p.dim())
        projected = []
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(p, i)
            #print(point.shape)
            tangent = self.take_submanifold_value(u, i)
            #print(tangent.shape)
            proj = manifold.proj_tan(tangent, point,c)
            #proj.requires_grad = True
            #print(proj.shape)
            proj = proj.reshape((*proj.shape[:target_batch_dim], -1))
            #print(proj.shape)
            #print("----------")
            
            projected.append(proj)

        res = torch.cat(projected, -1)
        #print(res.shape)
        #if not res.requires_grad:
        #    raise ValueError
        
        #print()

        #res.retain_grad()
        
        return res

    ##TODO: add zero funcationality to the 
    def proj_tan0(self, u, c):
        """Projects u on the tangent space of the origin."""
        target_batch_dim = _calculate_target_batch_dim(u.dim())
        projected = []
        #print("proj_tan0")
        #print(u.shape)
        for i, manifold in enumerate(self.manifolds):
            tangent = self.take_submanifold_value(u, i)
            #print(u == tangent)
            
            #print(tangent.shape)
            
            proj = manifold.proj_tan0(tangent,c)
            #print(proj.shape)
            #proj.requires_grad = True
            
            proj = proj.reshape((*proj.shape[:target_batch_dim], -1))
            #print(proj.shape)
            #print("----------")

            projected.append(proj)
        res = torch.cat(projected, -1)
        #print(res.shape)
        #if not res.requires_grad:
        #    raise ValueError

        res.retain_grad()

        return res



    def expmap(self, u, p, c):
        """Exponential map of u at point p."""
        target_batch_dim = _calculate_target_batch_dim(u.dim(), p.dim())
        #print("expmap")
        #print(u.shape)

        mapped_tensors = []
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(p, i)
            tangent = self.take_submanifold_value(u, i)
            mapped = manifold.expmap(tangent, point,c)
            #print(mapped)
            #mapped.requires_grad = True
            
            mapped = mapped.reshape((*mapped.shape[:target_batch_dim], -1))
            mapped_tensors.append(mapped)

        res = torch.cat(mapped_tensors, -1)

        #if not res.requires_grad:
        #    raise ValueError
        #print(res.shape)

        #res.retain_grad()
        
        return res


    def logmap(self, p1, p2, c):
        target_batch_dim = _calculate_target_batch_dim(p1.dim(), p2.dim())
        logmapped_tensors = []
        #print("logmap")
        #print(p1.shape)
        #print(p2.shape)
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(p1, i)
            point1 = self.take_submanifold_value(p2, i)
            logmapped = manifold.logmap(point, point1,c)
            #logmapped.requires_grad = True
            
            logmapped = logmapped.reshape((*logmapped.shape[:target_batch_dim], -1))
            #print(logmapped.shape)
            logmapped_tensors.append(logmapped)

        res = torch.cat(logmapped_tensors, -1)
        
        #print(res.shape)
        #if not res.requires_grad:
        #    raise ValueError


        #res.retain_grad()

        return res

    def expmap0(self, u, c):
        """Exponential map of u at point p."""
        #print("exp0")
        #print(u.shape)
        target_batch_dim = _calculate_target_batch_dim(u.dim())
        #print(u)
        mapped_tensors = []
        for i, manifold in enumerate(self.manifolds):
            tangent = self.take_submanifold_value(u, i)
            mapped = manifold.expmap0(tangent,c)
            #mapped.requires_grad = True
            
            #print(mapped)
            mapped = mapped.reshape((*mapped.shape[:target_batch_dim], -1))
            mapped_tensors.append(mapped)
        
        res = torch.cat(mapped_tensors, -1)
        
        #print(res.shape)
        #if not res.requires_grad:
        #    raise ValueError


        #res.retain_grad()

        return res



    def logmap0(self, p, c):
        target_batch_dim = _calculate_target_batch_dim(p.dim())
        logmapped_tensors = []
        #print("logmap0")
        #print(p.shape)
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(p, i)
            logmapped = manifold.logmap0(point,c)
            #logmapped.requires_grad = True
            
            logmapped = logmapped.reshape((*logmapped.shape[:target_batch_dim], -1))
            #print(logmapped.shape)
            logmapped_tensors.append(logmapped)


        res = torch.cat(logmapped_tensors, -1)
        #print(res.shape)
        #if not res.requires_grad:
        #    raise ValueError

        res.retain_grad()
        return res
    def mobius_add(self, x, y, c, dim=-1):
        target_batch_dim = _calculate_target_batch_dim(x.dim(), y.dim())
        transported_tensors = []
        #print("mobius add")
        #print(x.shape, y.shape)
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(x, i)
            point1 = self.take_submanifold_value(y, i)
            mob_add = manifold.mobius_add(point, point1, c)
            #mob_add.requires_grad = True
            
            mob_add = mob_add.reshape(
                (*mob_add.shape[:target_batch_dim], -1)
            )
            transported_tensors.append(mob_add)

        res = torch.cat(transported_tensors, -1)
        #res.retain_grad()
        
        #print(res.shape)
        return res        


    def mobius_matvec(self, m, x, c):

        """target_batch_dim = _calculate_target_batch_dim(m.dim(), x.dim())
        transported_tensors = []
        #print("mobius_matvec")
        #print(x.shape)
        #print(m.shape)
        for i, manifold in enumerate(self.manifolds):
            #point = m
            if m.shape[-1] == x.shape[-1]:
                point = self.take_submanifold_value(m, i, True)
            else: 
                point = self.take_submanifold_value(m, i, is_matvec = True)
                
            
            #print(x.shape)
            #point1 = x
            point1 = self.take_submanifold_value(x, i)
            #print(point.shape)
            #print(point1.shape)
            mob_matvec = manifold.mobius_matvec(point, point1, c)
            #mob_matvec.requires_grad = True
            
            mob_matvec = mob_matvec.reshape(
                (*mob_matvec.shape[:target_batch_dim], -1)
            )
            transported_tensors.append(mob_matvec)
        
        res =  torch.cat(transported_tensors, -1)

        #print(res.shape)
        return res"""


        target_batch_dim = _calculate_target_batch_dim(m.dim(), x.dim())
        transported_tensors = []
        #print("mobius_matvec")
        #print(x.shape)
        #print(m.shape)
        #res = m
        for i, manifold in enumerate(self.manifolds):
            #point = m
            #if m.shape[-1] == x.shape[-1]:
            point = self.take_submanifold_value(m, i, is_matvec = True)
            #else: 
            #    point = self.take_submanifold_value(m, i, is_matvec = True)
                
            
            #print(x.shape)
            #point1 = x
            point1 = self.take_submanifold_value(x, i)
            #print("point.shape", point.shape)
            #print(point1.shape)
            mob_matvec = manifold.mobius_matvec(point, point1, c)
            mob_matvec = mob_matvec.reshape(
                (*mob_matvec.shape[:target_batch_dim], -1)
            )
            #print(mob_matvec.shape)
            transported_tensors.append(mob_matvec)
        
        res =  torch.cat(transported_tensors, -1)
        #print(res.shape)
        #print("--------")
        return res



    def init_weights(self, w, c, irange=1e-5):
        """Initializes random weigths on the manifold."""
        target_batch_dim = _calculate_target_batch_dim(w.dim())
        randn = []
        for i, manifold in enumerate(self.manifolds):
            weight = self.take_submanifold_value(w, i)
            randed = manifold.init_weights(weight, c)
            #randed.requires_grad = True
            randed = randed.reshape((*randed.shape[:target_batch_dim], -1))
            randn.append(proj)
        return torch.cat(randn, -1)
        
        #return [man.init_weights(w[k], c[k]) for k, man in enumerate(self.manifolds)]
    

    def inner(self, p, c, u, v=None):
        """Inner product for tangent vectors at point x."""
        if v is not None:
            target_batch_dim = _calculate_target_batch_dim(p.dim(), u.dim(), v.dim())
        else:
            target_batch_dim = _calculate_target_batch_dim(p.dim(), u.dim())
        products = []
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(p, i)
            u_vec = self.take_submanifold_value(u, i)
            if v is not None:
                v_vec = self.take_submanifold_value(v, i)
            else:
                v_vec = None
            inner = manifold.inner(point,c, u_vec, v_vec, keepdim=True)
            #inner.requires_grad = True
            inner = inner.reshape(*inner.shape[:target_batch_dim], -1).sum(-1)
            products.append(inner)
            #print(inner)
        result = sum(products)
        if keepdim:
            result = torch.unsqueeze(result, -1)
        return result    


    def ptransp(self, x, y, u, c):
        target_batch_dim = _calculate_target_batch_dim(x.dim(), y.dim(), u.dim())
        transported_tensors = []
        #print("ptransp")
        #print(x.shape, y.shape, y.shape)
        for i, manifold in enumerate(self.manifolds):

            point = self.take_submanifold_value(x, i)
            point1 = self.take_submanifold_value(y, i)
            tangent = self.take_submanifold_value(u, i)
            transported = manifold.ptransp(point, point1, tangent,c)
            #transported.requires_grad = True
            
            transported = transported.reshape(
                (*transported.shape[:target_batch_dim], -1)
            )
            transported_tensors.append(transported)

        res = torch.cat(transported_tensors, -1)

        #print(res.shape)

        return res

    
    def take_submanifold_value(
            self, x: torch.Tensor, i: int, reshape=True, is_matvec = False    ) -> torch.Tensor:
        """
        Take i'th slice of the ambient tensor and possibly reshape.

        Parameters
        ----------
        x : tensor
            Ambient tensor
        i : int
            submanifold index
        reshape : bool
            reshape the slice?

        Returns
        -------
        torch.Tensor
        """
        #print(x.shape)
        #slc_length = int(x.shape[-1] / self.num_man)
        
        slc_length = int(x.shape[-1] / self.total_ratio)
        if is_matvec:
            slc_length_col = int(x.shape[-2] / self.total_ratio)
        #print(slc_length)
        slc = self.slices[i]
        #print(slc)
        #print(slc.start, slc.stop, slc.stop - slc.start)
        start = slc.start * slc_length
        length =  (slc.stop - slc.start) * slc_length
        if x.shape[-1] - (start + slc_length)< slc_length:
            length = x.shape[-1] - start

        if is_matvec:
            start_col = slc.start * slc_length_col
            length_col =  (slc.stop - slc.start) * slc_length_col
            if x.shape[-2] - (start_col + slc_length_col)< slc_length_col:
                length_col = x.shape[-2] - start_col

            #print(start_col, length_col)

        
        #part = x.narrow(-1, start, length)

        if not is_matvec:
            part = x.narrow(-1, start, length)
        else:
            part = torch.zeros((length_col,length)) + x[start_col:start_col+length_col, start:start+length]
        
        #print(part.shape)
        #print(part.shape)

        #if reshape:
        #    part = part.reshape((*part.shape[:-1], *self.shapes[i]))
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
        #res.retain_grad()
        return res

            
"""#the following has the reference: https://github.com/pymanopt/
class ndarraySequenceMixin:
    # The following attributes ensure that operations on sequences of
    # np.ndarrays with scalar numpy data types such as np.float64 don't attempt
    # to vectorize the scalar variable. Refer to
    #
    #     https://docs.scipy.org/doc/numpy/reference/arrays.classes.html
    #     https://github.com/pymanopt/pymanopt/issues/49
    #
    # for details.
    __array_priority__ = 1000
    __array_ufunc__ = None  # Available since numpy 1.13
    
class _ProductTangentVector(list, ndarraySequenceMixin):
    def __repr__(self):
        repr_ = super(_ProductTangentVector, self).__repr__()
        return "_ProductTangentVector: " + repr_

    def __add__(self, other):
        assert len(self) == len(other)
        return _ProductTangentVector(
            [v + other[k] for k, v in enumerate(self)])

    def __sub__(self, other):
        assert len(self) == len(other)
        return _ProductTangentVector(
            [v - other[k] for k, v in enumerate(self)])

    def __mul__(self, other):
        return _ProductTangentVector([other * val for val in self])

    __rmul__ = __mul__

    def __div__(self, other):
        return _ProductTangentVector([val / other for val in self])

    def __neg__(self):
        return _ProductTangentVector([-val for val in self])"""

