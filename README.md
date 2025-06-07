# Fourier Neural Operator (FNO)

## Introduction
The Fourier Neural Operator (FNO) is designed to solve Partial Differential Equation (PDE) problems in real‑world applications. This repository contains implementations of FNO focused on 1D and 2D PDE cases using PyTorch.

### Environment Setup
To set up your environment to use the Fourier Neural Operator, follow these steps:

1. Install the necessary packages using pip or conda:
   ```bash
   pip install -r requirements.txt
   conda install -r requirements.txt
   ```

> ### Notice for GPU Users
> Ensure your installed `torch` and `torchvision` versions match your CUDA toolkit. You can find the correct command for your CUDA version at https://pytorch.org/get-started/locally/.  
> If needed, uninstall and reinstall PyTorch with:

> ```bash
> pip uninstall torch torchvision
> pip install torch==<version>+cuXXX torchvision==<version>+cuXXX -f https://download.pytorch.org/whl/torch_stable.html
> ```

### Using FNO
Once the environment is set up, import the FNO module in your scripts or notebooks:
```python
from FNO_torch import FNOnd
# then instantiate and train your model
```

### Examples
Examples of execution are stored in `test_1d.ipynb` and `test_2d.ipynb`. These notebooks demonstrate how to apply the Fourier Neural Operator in 1D and 2D PDE problems with PyTorch.

> #### Dataset Availability
> Due to restrictions on GitLab, the datasets used here must be downloaded separately:
> - For `test_1d.ipynb`:  
>   https://ssd.mathworks.com/supportfiles/nnet/data/burgers1d/burgers_data_R10.mat  
> - For `test_2d.ipynb`:  
>   https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset  
> Please download these files and place them in the `data/` directory before running the examples.