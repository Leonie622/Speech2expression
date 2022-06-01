import torch
import numpy as np

n = 'pose.npy'

d = np.load(n)
np.save(n, -d)
