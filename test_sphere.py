import torch
import numpy as np

from manifolds.spherical import Spherical

S = Spherical()

x = np.random.rand(10, 3)
x = x / np.linalg.norm(x, axis=1)[:, None]
t = torch.tensor(x)

y = np.random.rand(10, 3)
p1 = S.mobius_matvec(S.logmap0(t, 2), torch.tensor(y), 2)
y = y / np.linalg.norm(y, axis=1)[:, None]
s = torch.tensor(y)
p2 = S.mobius_matvec(s, torch.tensor(y), 2)