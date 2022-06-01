import torch
import numpy as np

n = 'young.npy'

d = np.load(n)
np.save(n, -d)
print('ok')
