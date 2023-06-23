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
    def __init__(self, manifolds):
        super().__init__()
        # self.manifolds = manifolds
        self.manifolds = [x[0] for x in manifolds]
        self.name = "Product"

        self.n_manifolds = len(manifolds)

        start = 0
        self.slices = []
        self.num_spaces = 0
        for manifold, count in manifolds:
            end = start + count
            self.num_spaces += count
            self.slices.append(slice(start, end))
            start = end

    def sqdist(self, p1, p2, c):
        """Squared distance between pairs of points."""
        target_batch_dim = _calculate_target_batch_dim(p1.dim(), p2.dim())
        mini_dists2 = []
        print(p1, p2)
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
        return res

    def proj(self, p, c):
        """Projects point p on the manifold."""
        projected = []

        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(p, i)
            proj = manifold.proj(point,c)
            
            proj = proj.reshape(*p.shape[: len(p.shape) - 1], -1)
            projected.append(proj)

        res =  torch.cat(projected, -1)
        return res

    def proj_tan(self, u, p, c):
        """Projects u on the tangent space of p."""
        target_batch_dim = _calculate_target_batch_dim(u.dim(), p.dim())
        projected = []
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(p, i)
            tangent = self.take_submanifold_value(u, i)
            proj = manifold.proj_tan(tangent, point,c)
            proj = proj.reshape((*proj.shape[:target_batch_dim], -1))
            projected.append(proj)

        res = torch.cat(projected, -1)
        return res

    def proj_tan0(self, u, c):
        """Projects u on the tangent space of the origin."""
        target_batch_dim = _calculate_target_batch_dim(u.dim())
        projected = []
        for i, manifold in enumerate(self.manifolds):
            tangent = self.take_submanifold_value(u, i)
            proj = manifold.proj_tan0(tangent,c)
            proj = proj.reshape((*proj.shape[:target_batch_dim], -1))
            projected.append(proj)
        res = torch.cat(projected, -1)
        res.retain_grad()
        return res

    def expmap(self, u, p, c):
        """Exponential map of u at point p."""
        target_batch_dim = _calculate_target_batch_dim(u.dim(), p.dim())

        mapped_tensors = []
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(p, i)
            tangent = self.take_submanifold_value(u, i)
            mapped = manifold.expmap(tangent, point,c)
            mapped = mapped.reshape((*mapped.shape[:target_batch_dim], -1))
            mapped_tensors.append(mapped)

        res = torch.cat(mapped_tensors, -1)
        return res


    def logmap(self, p1, p2, c):
        target_batch_dim = _calculate_target_batch_dim(p1.dim(), p2.dim())
        logmapped_tensors = []
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(p1, i)
            point1 = self.take_submanifold_value(p2, i)
            logmapped = manifold.logmap(point, point1,c)
            logmapped = logmapped.reshape((*logmapped.shape[:target_batch_dim], -1))
            logmapped_tensors.append(logmapped)

        res = torch.cat(logmapped_tensors, -1)
        return res

    def expmap0(self, u, c):
        """Exponential map of u at point p."""
        target_batch_dim = _calculate_target_batch_dim(u.dim())
        mapped_tensors = []
        for i, manifold in enumerate(self.manifolds):
            tangent = self.take_submanifold_value(u, i)
            mapped = manifold.expmap0(tangent,c)
            mapped = mapped.reshape((*mapped.shape[:target_batch_dim], -1))
            mapped_tensors.append(mapped)
        
        res = torch.cat(mapped_tensors, -1)
        return res

    def logmap0(self, p, c):
        target_batch_dim = _calculate_target_batch_dim(p.dim())
        logmapped_tensors = []
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(p, i)
            logmapped = manifold.logmap0(point,c)
            logmapped = logmapped.reshape((*logmapped.shape[:target_batch_dim], -1))
            logmapped_tensors.append(logmapped)

        res = torch.cat(logmapped_tensors, -1)
        res.retain_grad()
        return res

    def mobius_add(self, x, y, c, dim=-1):
        target_batch_dim = _calculate_target_batch_dim(x.dim(), y.dim())
        transported_tensors = []
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(x, i)
            point1 = self.take_submanifold_value(y, i)
            mob_add = manifold.mobius_add(point, point1, c)
            mob_add = mob_add.reshape(
                (*mob_add.shape[:target_batch_dim], -1)
            )
            transported_tensors.append(mob_add)

        res = torch.cat(transported_tensors, -1)
        return res        

    def mobius_matvec(self, m, x, c):
        target_batch_dim = _calculate_target_batch_dim(m.dim(), x.dim())
        transported_tensors = []
        for i, manifold in enumerate(self.manifolds):
            # point = self.take_submanifold_value(m, i, is_matvec = True)
            point = self.take_submanifold_value(m, i)
            point1 = self.take_submanifold_value(x, i)
            print('point', point.size())
            print('point1', point1.size())
            mob_matvec = manifold.mobius_matvec(point, point1, c)
            mob_matvec = mob_matvec.reshape(
                (*mob_matvec.shape[:target_batch_dim], -1)
            )
            transported_tensors.append(mob_matvec)
        
        res =  torch.cat(transported_tensors, -1)
        return res

    def init_weights(self, w, c, irange=1e-5):
        """Initializes random weigths on the manifold."""
        target_batch_dim = _calculate_target_batch_dim(w.dim())
        randn = []
        for i, manifold in enumerate(self.manifolds):
            weight = self.take_submanifold_value(w, i)
            randed = manifold.init_weights(weight, c)
            randed = randed.reshape((*randed.shape[:target_batch_dim], -1))
            randn.append(randed)
        return torch.cat(randn, -1)

    def inner(self, p, c, u, v=None, keepdim=True):
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
            inner = inner.reshape(*inner.shape[:target_batch_dim], -1).sum(-1)
            products.append(inner)
        result = sum(products)
        if keepdim:
            result = torch.unsqueeze(result, -1)
        return result    

    def ptransp(self, x, y, u, c):
        target_batch_dim = _calculate_target_batch_dim(x.dim(), y.dim(), u.dim())
        transported_tensors = []
        for i, manifold in enumerate(self.manifolds):

            point = self.take_submanifold_value(x, i)
            point1 = self.take_submanifold_value(y, i)
            tangent = self.take_submanifold_value(u, i)
            transported = manifold.ptransp(point, point1, tangent,c)
            transported = transported.reshape(
                (*transported.shape[:target_batch_dim], -1)
            )
            transported_tensors.append(transported)

        res = torch.cat(transported_tensors, -1)
        return res
    
    def take_submanifold_value(self, x, i, reshape=True, is_matvec=False):
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

    # def take_submanifold_value(
    #     self, x: torch.Tensor, i: int, reshape=True
    # ) -> torch.Tensor:
    #     """
    #     Take i'th slice of the ambient tensor and possibly reshape.

    #     Parameters
    #     ----------
    #     x : tensor
    #         Ambient tensor
    #     i : int
    #         submanifold index
    #     reshape : bool
    #         reshape the slice?

    #     Returns
    #     -------
    #     torch.Tensor
    #     """
    #     slc = self.slices[i]
    #     part = x.narrow(-1, slc.start, slc.stop - slc.start)
    #     # if reshape:
    #     #     part = part.reshape((*part.shape[:-1], *self.shapes[i]))
    #     print('part', part.size())
    #     return part


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
