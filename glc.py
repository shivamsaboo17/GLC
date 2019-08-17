import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import Subset
import numpy as np
import torch.nn.functional as F
from datasets import *

class CorrectionGenerator:
    
    def __init__(self, simulate=True, trusted_dataset=None,
                untrusted_dataset=None, dataset=None, randomization_strength=None):
        if not simulate and ((trusted_dataset is None) or (untrusted_dataset is None)):
            raise ValueError('Provide trusted and untrusted datasets')
        if simulate and (dataset is None) or (randomization_strength is None):
            raise ValueError('Cannot simulate without dataset and randomization strength')
        if not simulate:
            self.trusted_dataset = trusted_dataset
            self.untrusted_dataset = untrusted_dataset
        else:
            self.prepare_datasets(dataset)
        self.dataset_dicts = self.prepare_class_generators(self.trusted_dataset)
        self.untrusted_dataset = RandomizedDataset(self.untrusted_dataset, len(self.dataset_dicts.keys()),
                                                  randomization_strength)
        
    def fetch_datasets(self):
        return self.trusted_dataset, self.untrusted_dataset
    
    def generate_correction_matrix(self, noisy_model, batch_size):
        return self.build_label_correction_matrix(noisy_model, self.dataset_dicts, batch_size)
    
    def random_true_noisy_split(self, dataset, true_rat=0.1):
        true_idx = int(true_rat * len(dataset))
        idxs = np.arange(0, len(dataset))
        np.random.shuffle(idxs)
        return Subset(dataset, idxs[:true_idx]), Subset(dataset, idxs[true_idx:])
    
    def prepare_datasets(self, dataset, trusted_rat=0.1):
        self.trusted_dataset, self.untrusted_dataset = self.random_true_noisy_split(dataset, trusted_rat)
        
    def prepare_indices(self, dataset):
        index_enabled_dataset = IndexEnabledDataset(dataset)
        dl = DataLoader(index_enabled_dataset, batch_size=1)
        indices = defaultdict(list)
        for (x, y, index) in tqdm(dl):
            indices[int(y[0].data)].append(int(index[0].data))
        return indices
    
    def prepare_class_generators(self, dataset):
        indices = self.prepare_indices(dataset)
        return {k:Subset(dataset, indices[k]) for k in indices.keys()}
    
    def build_label_correction_matrix(self, noisy_model, clean_ds_dicts, batch_size=32):
        num_labels = len(clean_ds_dicts.keys())
        label_correction_matrix = torch.zeros((num_labels, num_labels))
        for lab, idx in enumerate(clean_ds_dicts):
            clean_dl = DataLoader(clean_ds_dicts[idx], batch_size=batch_size)
            pbar = tqdm(clean_dl)
            pbar.set_description(f'Processing label {lab}')
            for data, target in pbar:
                predicted_proba = F.softmax(noisy_model(data), dim=1).mean(dim=0)
                label_correction_matrix[idx, :] += predicted_proba
            label_correction_matrix[idx, :] = label_correction_matrix[idx, :] / len(clean_dl)
        print('Done')
        return label_correction_matrix

class GoldCorrectionLossFunction(nn.Module):
    def __init__(self, label_correction_matrix):
        super(GoldCorrectionLossFunction, self).__init__()
        self.label_correction_matrix = label_correction_matrix.data
    def forward(self, x, y):
        c_loss = nn.CrossEntropyLoss(reduction='none')(x[0], y) * x[1].data.float()
        n_loss = nn.NLLLoss(reduction='none')(torch.log(torch.matmul(F.softmax(x[0], dim=1), self.label_correction_matrix.data)),
                                 y) * (1 - x[1]).data.float()
        return c_loss.mean() + n_loss.mean()