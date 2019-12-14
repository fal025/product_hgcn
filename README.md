CS229 project - product space GCN
==================================================

Adapted from https://github.com/HazyResearch/hgcn 

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

For now, the choice of product space should be the ratio between three spaces <E,S,P> corresponding to Euclidean, Spherical and Hyperboloid(PoincareBall) and the sum should be a divisor of your dimension input. An example input would be `P1S1E2`, where 4 divids 16.
The dataset name for now can be chosen from `cora` for `nc` and `lp`, `pubmed` for `nc` (or `lp` or if one have memory > 32G due to the large number of nodes in the dataset; otherwise the initialization step ), `disease_lp` for `lp`, `disease_nc` for `nc`, `airport` for `lp` and `nc`. 

The result can be replicate through the running the `run_test.py` script.

