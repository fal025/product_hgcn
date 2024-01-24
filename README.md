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

