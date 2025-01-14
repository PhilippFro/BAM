# Graph Structure Inference with BAM: Neural Dependency Processing via Bilinear Attention

This repository contains the official implementation of the NeurIPS 2024 paper ["Graph Structure Inference with BAM: Neural Dependency Processing via Bilinear Attention"](https://neurips.cc/virtual/2024/poster/96766).

## Authors
- Philipp Froehlich (TU Darmstadt)
- Heinz Koeppl (TU Darmstadt)

## Overview
BAM (Bilinear Attention Mechanism) is a novel approach to graph structure inference that leverages neural attention mechanisms to process dependencies in data. Our method combines observational attention between features and samples with bilinear attention in the SPD space to effectively learn and identify graph structures.

## Installation

### Dependencies
The code requires Python 3.10.8 and the following packages:
- TensorFlow 
- NumPy
- Pandas

### Quick Setup
1. Clone this repository:
```
git clone https://github.com/PhilippFro/BAM.git
cd BAM
```

2. Create a conda environment using the provided environment file:
```
conda env create -f BAM_env.yml
conda activate BAM_env
```

## Usage

### Training the Model
To train the BAM model with default settings:
```
python edge_classifier_BAM.py
```

### Model Configuration
The main model parameters can be configured in `edge_classifier_BAM.py`:

```
model = cl.model_attention_final(
    n_channels_main=100,  # Number of channels in main layer
    data_layers=10,       # Number of layers for attention between features and samples
    cov_layers=10,        # Number of layers for bilinear attention
    inner_channels=100,   # Number of inner channels
    N_exp=3,             # Number of exponentiations for SPD activation
    N_heads=5            # Number of attention heads
)
```

### Training Parameters
You can modify the training parameters in `edge_classifier_BAM.py`:

```
spe = 128      # Steps per epoch
ep = 1000      # Number of epochs
N = 1          # Batch size
M_min = 50     # Minimum number of samples
M_max = 1000   # Maximum number of samples
d_min = 10     # Minimum dimension
d_max = 100    # Maximum dimension
```

## Repository Structure
- `custom_layers_BAM.py`: Implementation of BAM model architecture
- `edge_classifier_BAM.py`: Main training script
- `generator_cheby_BAM.py`: Data generation utilities
- `helpers_BAM.py`: Helper functions for training and evaluation
- `testing_generator_BAM.py`: Testing utilities
- `BAM_env.yml`: Conda environment specification

## Citation
If you find this code useful in your research, please cite our paper:

```
@inproceedings{froehlich2024bam,
    title={Graph Structure Inference with {BAM}: Neural Dependency Processing via Bilinear Attention},
    author={Froehlich, Philipp and Koeppl, Heinz},
    booktitle={Advances in Neural Information Processing Systems ({NeurIPS})},
    volume={36},
    year={2024},
    url={https://neurips.cc/virtual/2024/poster/96766}
}
```

## License
MIT License

Copyright (c) 2024 Philipp Froehlich, Heinz Koeppl

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
