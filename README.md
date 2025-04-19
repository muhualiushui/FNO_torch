# Fourier Neural Operator (FNO)

## Introduction
The Fourier Neural Operator (FNO) is designed to solve Partial Differential Equation (PDE) problems in real-world applications. This repository contains implementations of FNO focused on 1D and 2D PDE cases.

### Environment Setup
To set up your environment to use the Fourier Neural Operator, follow these steps:

Install the necessary packages using pip:

   ```bash
   pip install -r requirements.txt
   
   ```

> ### Notice for GPU Users
> The `requirements.txt` includes specific CUDA drivers compatible with Jax. If your system's CUDA version does not support the driver specified in the file, you will need to modify the driver version in the `requirements.txt`. Ensure that you maintain the same package versions as the new driver and that these changes do not affect the functionality of the code.

### Using FNO
Once the environment is set up, you can use the FNO implementation by importing the FNO module. Be sure to pay close attention to each parameter within the FNO module, as explanations are provided within the code to assist in understanding their functions and impacts.

### Examples
Examples of execution are stored in the `test_1d.ipynb` and `test_2d.ipynb` for reference. These examples demonstrate how to apply the Fourier Neural Operator in 1D and 2D PDE problems. You can better understand the practical application of the FNO by reviewing these examples.

> #### Dataset Availability:
> Due to restrictions on GitLab, the dataset used for the repository can be found in the following URL:
> ##### For `test_1d.ipynb`:
> https://ssd.mathworks.com/supportfiles/nnet/data/burgers1d/burgers_data_R10.mat
> ##### For `test_2d.ipynb`:
> [https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset)
> Please download the dataset from this link to properly run the `test_2d.ipynb` example.

# 傅里叶神经算子 (FNO)

## 简介
傅里叶神经算子（FNO）设计用于解决现实世界中的偏微分方程（PDE）问题。本仓库包含了专注于1D和2D PDE案例的FNO实现。

### 环境设置
要设置您的环境以使用傅里叶神经算子，请按照以下步骤操作：

使用pip安装必要的包：

   ```bash
   pip install -r requirements.txt
   ```

> ### GPU用户须知
> `requirements.txt`包含了与Jax兼容的特定CUDA驱动程序。如果您的系统的CUDA版本不支持文件中指定的驱动程序，您将需要在`requirements.txt`中修改驱动程序版本。确保您保持与新驱动程序相同的包版本，并且这些更改不会影响代码的功能。

### 使用FNO
环境设置完成后，您可以通过导入FNO模块来使用FNO实现。请仔细关注FNO模块中的每个参数，代码中提供了解释，以帮助理解它们的功能和影响。

### 示例
示例代码保存在 `test_1d.ipynb` 和 `test_2d.ipynb` 中供参考。这些示例展示了如何在1D和2D偏微分方程问题中应用傅里叶神经算子。通过查看这些示例，您可以更好地理解FNO的实际应用。

> #### 数据集可用性：
> 由于GitLab的限制，用于该仓库的数据集可以在以下URL中找到：
> ##### 对于 `test_1d.ipynb`：
> https://ssd.mathworks.com/supportfiles/nnet/data/burgers1d/burgers_data_R10.mat
> ##### 对于 `test_2d.ipynb`：
> [https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset)
> 请从这些链接下载数据集，以正确运行 `test_2d.ipynb` 示例。
