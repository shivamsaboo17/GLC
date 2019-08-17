# GLC
Unofficial implementation of [Using Trusted Data to Train Deep Networks on
Labels Corrupted by Severe Noise](https://arxiv.org/pdf/1802.05300.pdf) (NIPS 18) in PyTorch. Gold Loss Correction is implemented for training neural networks with labels corrupted with severe noise.

## Usage
```python
from datasets import GoldCorrectionDataset
from glc import CorrectionGenerator, GoldCorrectionLossFunction

c_gen = CorrectionGenerator(simulate=True, dataset=trn_ds, randomization_strength=1.0)

# Fetch both corrupted and clean datasets if in simuate mode
trusted_dataset, untrusted_dataset = c_gen.fetch_datasets()

"""
Train the model on untrusted_dataset
"""
# Generate correction matrix
label_correction_matrix = c_gen.generate_correction_matrix(trainer.model, 32)

# Wrap trusted and untrusted dataset together using GoldCorrectionDataset class
gold_ds = GoldCorrectionDataset(trusted_dataset, untrusted_dataset)
gold_dl = DataLoader(gold_ds, batch_size=32, shuffle=True)

# Modified loss function
gold_loss = GoldCorrectionLossFunction(label_correction_matrix)

"""
Train using gold_ds and gold_loss the model, until convergence
"""
```

## Results
### MNIST
#### Regular training on trusted data (~5% of entire data) -> 61.12 accuracy
#### Gold Loss Correction with 5% trusted -> 95.45 accuracy (All samples in untrusted data (95% of total data) is corrupted by randomly assigning labels)
