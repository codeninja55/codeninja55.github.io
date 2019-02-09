---
layout: post
title: Part 1 of 3 - Building a Deep Learning PC
status: draft
---

This is a three part series that I will endeavour to update regularly as I discover better workflows.
1. Part 1 (this post): Hardware Drivers and System OS Installation.
2. Part 2: Development Environment and Library Installation
3. Part 3: Configuring Remote Access and Testing your new Development Environment 

## Hardware and System OS Installation
Just a reminder these are the hardware and software configurations I have been using as part of this series.

#### Hardware Configurations
```
CPU: Intel Core i9 9900K
Motherboard: Asus ROG Maximus XI Extreme
Memory: 32GB DIMM DDR4 3200MHZ
Hard Drive: 2x Samsung 970 Pro 521GB V-NAND NVMe M.2 SSD*
Additional Persistent Storage: 6x Samsung 860 Pro 1TB V-NAND SATA SSD in Raid 0 with Intel Rapid Storage**
Onboard GPU: Intel UHD Graphics support
Additional GPU: 2x Asus GeForce RTX 2080 Ti Dual with nVidia NVLink
```
\* Windows 10 was installed in one SSD to allow easier configurations of BIOS with utilities provided by Asus.

** There is still some considerations to use Intel Optane Memory Raid however, there are some concerns with PCI-e lanes 
with adding too many devices in addition to running dual GPU (will update further). 

#### System Configurations
```
Linux Kernel: 4.20.0-042000-generic*
OS Version: Ubuntu 18.04.1 LTS Bionic
GPU Driver: nvidia-driver-415*
CUDA Version: CUDA 10.0*
```
\* These are the latest versions as of this post 2018-12-27.





## Installing Virtual Environment (Anaconda)
When managing software development environments, particularly those with complex libraries such those used for machine 
and deep learning, I prefer to use [Anaconda](https://www.anaconda.com/what-is-anaconda/). You could also use [miniconda](https://conda.io/miniconda.html) - a more lightweight version of Anaconda.

```bash
$ cd ~/Downloads
$ wget https://repo.continuum.io/archive/Anaconda3-2018.12-Linux-x86_64.sh
$ chmod +x ./Anaconda3-2018.12-Linux-x86_64.sh
$ ./Anaconda3-2018.12-Linux-x86_64.sh
```

### Creating the deep learning environment
To create an environment with Anaconda, you will generally use the `conda` command in a terminal. To install packages 
for this environment, you will also use either `conda install` or `pip install`.

```bash
$ conda create --name deep_learning_cv python=3.6 -c anaconda
$ source activate deep_learning_cv
$ conda env list
# conda environments:
#
base                     /home/codeninja/anaconda3
deep_learning_cv      *  /home/codeninja/anaconda3/envs/deep_learning_cv
```
Some notes to consider:
* The `base` environment is the conda environment with the default Python (in this case, Python 3.7).
* Using `source activate deep_learning_cv` will enter the you created and anything you install with `conda` will be 
added to this environment. 
* The asterisk `*` from the `conda env list` output is the currently activated environment. 
* From this point onwards, all commands marked `(deep_learning_cv) $ ` will mean the command was run with the activated environment.

### Installing deep learning libraries
In this short tutorial, to prepare for all situations, we will be installing a wide range of deep learning and computer 
vision libraries, including:
* OpenCV
* Tensorflow
* Keras
* PyTorch
* MxNet

```bash
(deep_learning_cv) $ conda install -c anaconda tensorflow-gpu mxnet opencv keras pytorch

## Package Plan ##

  environment location: /home/codeninja/anaconda3/envs/deep_learning_cv

  added / updated specs: 
    - keras
    - mxnet
    - opencv
    - pytorch
    - tensorflow-gpu


The following packages will be downloaded:
...
```
