Mixed-Curvature Product Space GCN
==================================================

Adapted from https://github.com/HazyResearch/hgcn 
and further forked from https://github.com/fal025/product_hgcn.

The dependencies are the following:

```
virtualenv -p [PATH to python3.7 binary] hgcn
source hgcn/bin/activate
pip install -r requirements.txt
```

To run additional tests, please use `train.py`
This script trains models for link prediction and node classification tasks. Metrics are printed at the end of training or can be saved in a directory by adding the command line argument `--save=1`.

For our purposes, the following command may be more useful than others:

```
python3 train.py --task lp --dataset <dataset_name> --model HGCN --lr 0.01 --dim <num_dim> --num-layers 2 --act relu --bias 0 --dropout 0.5 --weight-decay 0.001 --manifold <choice_of_prod_space> --log-freq 5 --cuda 0 --c 1
```

To specify a choice of product space use a string of the form <H{x}E{y}S{z}> x, y, and z indicate the number of copies of Hyperbolic, Euclidean, and Spherical space, respectively. An example input would be `H1E2S2`.

## Citation
The modified Mixed-Curvature Product Space GCN is described in the following preprint:  
[Product Manifold Representations for Learning on Biological Pathways](https://arxiv.org/abs/2401.15478)  
Daniel McNeela, Frederic Sala<sup>+</sup>, Anthony Gitter<sup>+</sup>.  
arXiv:2401.15478. 2024

See also the [hgcn repository references](https://github.com/HazyResearch/hgcn?tab=readme-ov-file#references).

## License
The code is available under the [Apache License 2.0](LICENSE).
Most of the source code is derived from the unlicensed [hgcn](https://github.com/HazyResearch/hgcn) and [product_hgcn](https://github.com/fal025/product_hgcn) repositories.
The named contributors of those repositories have been added to the license copyright.

The hgcn repository notes additional code was forked from the following repositories

 * [pygcn](https://github.com/tkipf/pygcn)
 * [gae](https://github.com/tkipf/gae)
 * [hyperbolic-image-embeddings](https://github.com/KhrulkovV/hyperbolic-image-embeddings)
 * [pyGAT](https://github.com/Diego999/pyGAT)
 * [poincare-embeddings](https://github.com/facebookresearch/poincare-embeddings)
 * [geoopt](https://github.com/geoopt/geoopt)
