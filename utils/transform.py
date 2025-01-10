import numpy as np
import torch

class Normalizer(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
    
    def __call__(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        return (x - self.mean) / self.std
        
