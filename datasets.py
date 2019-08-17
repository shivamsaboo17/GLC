import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import Subset
import numpy as np
import torch.nn.functional as F

class IndexEnabledDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index
    def __len__(self):
        return len(self.dataset)
    
class RandomizedDataset(Dataset):
    def __init__(self, dataset, num_classes, p=0.5, mode='random_uniform'):
        self.dataset = dataset
        self.random_offsets = []
        for i in range(len(self.dataset)):
            if np.random.uniform() < p:
                self.random_offsets.append(np.random.randint(0, num_classes))
            else:
                self.random_offsets.append(0)
        self.mode = mode
        self.num_classes = num_classes
        self.p = p
    def __getitem__(self, index):
        data, target = self.dataset[index]
        if self.mode == 'random_uniform':
            target = (target + self.random_offsets[index]) % self.num_classes
        return data, target
    def __len__(self):
        return len(self.dataset)
    
class GoldCorrectionDataset(Dataset):
    def __init__(self, true_dataset, noisy_dataset, valid=False):
        self.true_dataset = true_dataset
        self.noisy_dataset = noisy_dataset
        self.valid = valid
    def __getitem__(self, index):
        if index < len(self.true_dataset):
            x, y = self.true_dataset[index]
            return [x, 1], y
        x, y = self.noisy_dataset[index - len(self.true_dataset)]
        if self.valid:
            return [x, 1], y
        return [x, 0], y
    def __len__(self):
        return len(self.true_dataset) + len(self.noisy_dataset)