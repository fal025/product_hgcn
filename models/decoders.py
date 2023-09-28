"""Graph decoders."""
import manifolds
import torch.nn as nn
import torch.nn.functional as F

from layers.att_layers import GraphAttentionLayer
from layers.layers import GraphConvolution, Linear


class Decoder(nn.Module):
    """
    Decoder abstract class for node classification tasks.
    """

    def __init__(self, c):
        super(Decoder, self).__init__()
        self.c = c

    def decode(self, x, adj):
        #print("decoder super")
        if self.decode_adj:
            input = (x, adj)
            probs, _ = self.cls.forward(input)
        else:
            probs = self.cls.forward(x)
        return probs


class GCNDecoder(Decoder):
    """
    Graph Convolution Decoder.
    """

    def __init__(self, c, args):
        super(GCNDecoder, self).__init__(c)
        act = lambda x: x
        self.cls = GraphConvolution(args.dim, args.n_classes, args.dropout, act, args.bias)
        self.decode_adj = True


class GATDecoder(Decoder):
    """
    Graph Attention Decoder.
    """

    def __init__(self, c, args):
        super(GATDecoder, self).__init__(c)
        self.cls = GraphAttentionLayer(args.dim, args.n_classes, args.dropout, F.elu, args.alpha, 1, True)
        self.decode_adj = True


class LinearDecoder(Decoder):
    """
    MLP Decoder for Hyperbolic/Euclidean node classification models.
    """

    def __init__(self, c, args):
        super(LinearDecoder, self).__init__(c)

        if args.manifold not in ["Spherical", "Euclidean", "PoincareBall", "Hyperboloid"]:
            manifold_array = []
            word = list(args.manifold)
            for i in range(0,len(word), 2):
                if word[i] == "E":
                    man_name = "Euclidean"
                elif word[i] == "P":
                    man_name = "PoincareBall"
                elif word[i] == "S":
                    man_name = "Spherical"
                elif word[i] == "H":
                    man_name = "Hyperboloid"
                else:
                    raise ValueError("Invalide string in the manifold")
                count = int(word[i+1])
                #for j in range(count):
                manifold_array.append((getattr(manifolds, man_name)(),count))
                    
            self.manifold_name = "productManifold"
            self.manifold = getattr(manifolds, self.manifold_name)(manifold_array,args.dim)

        else:
            self.manifold = getattr(manifolds, args.manifold)()

        #self.manifold = getattr(manifolds, args.manifold)()
        self.input_dim = args.dim
        self.output_dim = args.n_classes
        self.bias = args.bias
        self.cls = Linear(self.input_dim, self.output_dim, args.dropout, lambda x: x, self.bias)
        self.decode_adj = False

    def decode(self, x, adj):
        h = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.c), c=self.c)
        return super(LinearDecoder, self).decode(h, adj)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, c={}'.format(
                self.input_dim, self.output_dim, self.bias, self.c
        )


model2decoder = {
    'GCN': GCNDecoder,
    'GAT': GATDecoder,
    'HNN': LinearDecoder,
    'HypGCN': LinearDecoder,
    'MLP': LinearDecoder,
    'Shallow': LinearDecoder,
}

