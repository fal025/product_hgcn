"""Graph encoders."""
import manifolds
import layers.hyp_layers as hyp_layers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.math_utils as pmath
from layers.att_layers import GraphAttentionLayer
from layers.layers import GraphConvolution, Linear, get_dim_act


class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    def __init__(self, c):
        super(Encoder, self).__init__()
        self.c = c

    def encode(self, x, adj):
        if self.encode_graph:
            input = (x, adj)
            output, _ = self.layers.forward(input)
        else:
            output = self.layers.forward(x)
        return output

class MLP(Encoder):
    """
    Multi-layer perceptron.
    """

    def __init__(self, c, args):
        super(MLP, self).__init__(c)
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            layers.append(Linear(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*layers)
        self.encode_graph = False


class HNN(Encoder):
    """
    Hyperbolic Neural Networks.
    """

    def __init__(self, c, args):
        super(HNN, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        assert args.num_layers > 1
        dims, acts, _ = hyp_layers.get_dim_act_curv(args)
        hnn_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hnn_layers.append(
                    hyp_layers.HNNLayer(
                            self.manifold, in_dim, out_dim, self.c, args.dropout, act, args.bias)
            )
        self.layers = nn.Sequential(*hnn_layers)
        self.encode_graph = False

    def encode(self, x, adj):
        x_hyp = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c), c=self.c)
        return super(HNN, self).encode(x_hyp, adj)

class GCN(Encoder):
    """
    Graph Convolution Networks.
    """

    def __init__(self, c, args):
        super(GCN, self).__init__(c)
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        gc_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            gc_layers.append(GraphConvolution(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*gc_layers)
        self.encode_graph = True


class HypGCN(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, c, args):
        super(HypGCN, self).__init__(c)
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
                manifold_array.append((getattr(manifolds, man_name)(),count))
            self.manifold_name = "productManifold"
            self.manifold = getattr(manifolds, self.manifold_name)(manifold_array, args.dim)

                    
        else:
            self.manifold = getattr(manifolds, args.manifold)()
        
        assert args.num_layers > 1
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        self.curvatures.append(self.c)
        hgc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                    hyp_layers.HyperbolicGraphConvolution(
                            self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias, args.use_att
                    )
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True

    def encode(self, x, adj):
        x_hyp = x
        # x_hyp = self.manifold.proj(
        #         self.manifold.expmap0(self.manifold.proj_tan0(x, self.curvatures[0]), c=self.curvatures[0]),
        #         c=self.curvatures[0])
        return super(HypGCN, self).encode(x_hyp, adj)


class GAT(Encoder):
    """
    Graph Attention Networks.
    """

    def __init__(self, c, args):
        super(GAT, self).__init__(c)
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        gat_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            assert dims[i + 1] % args.n_heads == 0
            out_dim = dims[i + 1] // args.n_heads
            concat = True
            gat_layers.append(
                    GraphAttentionLayer(in_dim, out_dim, args.dropout, act, args.alpha, args.n_heads, concat))
        self.layers = nn.Sequential(*gat_layers)
        self.encode_graph = True


class Shallow(Encoder):
    """
    Shallow Embedding method.
    Learns embeddings or loads pretrained embeddings and uses an MLP for classification.
    """

    def __init__(self, c, args):
        super(Shallow, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.use_feats = args.use_feats
        weights = torch.Tensor(args.n_nodes, args.dim)
        if not args.pretrained_embeddings:
            weights = self.manifold.init_weights(weights, self.c)
            trainable = True
        else:
            weights = torch.Tensor(np.load(args.pretrained_embeddings))
            assert weights.shape[0] == args.n_nodes, "The embeddings you passed seem to be for another dataset."
            trainable = False
        self.lt = manifolds.ManifoldParameter(weights, trainable, self.manifold, self.c)
        self.all_nodes = torch.LongTensor(list(range(args.n_nodes)))
        layers = []
        if args.pretrained_embeddings is not None and args.num_layers > 0:
            # MLP layers after pre-trained embeddings
            dims, acts = get_dim_act(args)
            if self.use_feats:
                dims[0] = args.feat_dim + weights.shape[1]
            else:
                dims[0] = weights.shape[1]
            for i in range(len(dims) - 1):
                in_dim, out_dim = dims[i], dims[i + 1]
                act = acts[i]
                layers.append(Linear(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*layers)
        self.encode_graph = False

    def encode(self, x, adj):
        h = self.lt[self.all_nodes, :]
        if self.use_feats:
            h = torch.cat((h, x), 1)
        return super(Shallow, self).encode(h, aui)


class ProdGCN(Encoder):
    """
    Product-GCN.
    """

    def __init__(self, c, args):
        super(ProdGCN, self).__init__(c)
        print(args.manifold)
        print(manifolds)
        self.manifold = getattr(manifolds, args.manifold)()
        assert args.num_layers > 1
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        self.curvatures.append(self.c)
        hgc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                    hyp_layers.HyperbolicGraphConvolution(
                            self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias, args.use_att
                    )
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True

    def encode(self, x, adj):
        x_hyp = self.manifold.proj(
                self.manifold.expmap0(self.manifold.proj_tan0(x, self.curvatures[0]), c=self.curvatures[0]),
                c=self.curvatures[0])
        return super(ProdGCN, self).encode(x_hyp, adj)
