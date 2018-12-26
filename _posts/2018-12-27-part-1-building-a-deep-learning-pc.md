---
layout: post
title: Part 1 of 3 - Building a Deep Learning PC
---

So while I have only recently slowly got into data science and using deep learning libraries. I have been preparing for 
this future ever since I started my computer science degree two years ago. And now it has come the time I really 
(yes, really) needed to get myself some better hardware to move to the next level of my education.

Machine learning and deep learning optimized computers are an expensive endeavour. Trust me, I have looked (if you are 
curious, this [prebuilt device](https://www.nvidia.com/en-au/data-center/dgx-2/) by nVidia is pretty slick). While my 
system is worth more than what most people would spend on a PC, I reasoned that I would using this to build some fairly
large ML models before long and would use it going forward into my academic research career. So I was more than willing
to add some serious juice. 

This is a three part series that I will endeavour to update regularly as I discover better workflows.
1. Part 1 (this post): Hardware and System OS Installation.
2. Part 2: Configuring Remote External Access
3. Part 3: Testing your new Development Environment 

## Hardware and System OS Installation
#### Hardware Configurations
```
CPU: Intel Core i9 9900K
Motherboard: Asus ROG Maximus XI Extreme
Memory: 32GB DIMM DDR4 3200MHZ
Hard Drive: 2x Samsung 970 Pro 521GB V-NAND NVMe M.2 SSD*
Additional Storage: 6x Samsung 860 Pro 1TB V-NAND SATA SSD in Raid 0 with Intel Rapid Storage**
Onboard GPU: Intel UHD Graphics support
Additional GPU: 2x Asus GeForce RTX 2080 Ti Dual with nVidia NVLink
```
\* Windows 10 was installed in one SSD to allow easier configurations of BIOS with utilities provided by Asus.

** There is still some considerations to use Intel Optane Raid however, there are some concerns with PCI-e lanes with adding too many devices in addition to running dual GPU (will update further). 

#### System Configurations
```
Linux Kernel: 4.20.0-042000-generic*
OS Version: Ubuntu 18.04.1 LTS Bionic
GPU Driver: nvidia-driver-415*
CUDA Version: CUDA 10.0*
```
* These are the latest versions as of this post 2018-12-27.

### Pre-Installation System Settings
* Use F8 to enter one-time boot menu 
* Install Windows 10 to use the Asus EZUpdate provided utility for quick BIOS updates.
* Press F2 when you see the Republic of Gamers logo to enter BIOS settings.
* Ensure `Secure Boot` is set to Windows UEFI.
* Disable `Fast Boot`
* Download from [here](http://releases.ubuntu.com/18.04/) or directly from 
[ubuntu-18.04.1-desktop-amd64.iso](http://releases.ubuntu.com/18.04/ubuntu-18.04.1-desktop-amd64.iso) and create a 
bootable USB with [Win32DiskImager](https://sourceforge.net/projects/win32diskimager/).

### Installing Ubuntu
When installing Ubuntu, ensure you are installing the full version and not the minimal version. I have had troubles with
these configurations in the past. Also ensure you are installing Ubuntu 18.04 on the SSD that is empty, not the one that you used to install Windows 10. Usually this will be either `nvme0n1` or `nvme1n1`.

Additionally, I prefer to encrypt my Ubuntu OS whenever I use it so I also selected 'LVM' and 'Encryption'.

Once Ubuntu has finished installing, the first thing you will need to do is update the system to the latest versions of packages. To do this, run this in a terminal.

```bash
$ sudo apt update && sudo apt dist-upgrade 
```

## Setting up the Ubuntu 18.04 for Hardware Compatibility
At this stage, we want to set up the Ubuntu to ensure all packages required for deep learning and GPU support will be ready. To do this, we will also be installing some additional useful packages.

### Git and Git LFS
Version control and large file control for large datasets that will be used. 

```bash
$ sudo apt install -y git
$ git config --global user.name "your-user-name"
$ git config --global user.email "your-github-email"
```

To install Git LFS, you will need to copy these commands:
```bash
$ sudo apt install -y gnupg apt-transport-https
$ curl -L https://packagecloud.io/github/git-lfs/gpgkey | sudo apt-key add -
$ deb https://packagecloud.io/github/git-lfs/ubuntu/ $(lsb_release -cs) main
$ deb-src https://packagecloud.io/github/git-lfs/ubuntu/ $(lsb_release -cs) main
$ sudo apt update
$ sudo apt install -y git-lfs
```


## Installing Virtual Environment (Anaconda)
When managing software development environments, particularly those with complex libraries such those used for machine and deep learning, I prefer to use [Anaconda](https://www.anaconda.com/what-is-anaconda/). You could also use [miniconda](https://conda.io/miniconda.html) - a more lightweight version of Anaconda.

```bash
$ cd ~/Downloads
$ wget https://repo.continuum.io/archive/Anaconda3-2018.12-Linux-x86_64.sh
$ chmod +x ./Anaconda3-2018.12-Linux-x86_64.sh
$ ./Anaconda3-2018.12-Linux-x86_64.sh
```

### Creating the deep learning environment
To create an environment with Anaconda, you will generally use the `conda` command in a terminal. To install packages for this environment, you will also use either `conda install` or `pip install`.

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
* Using `source activate deep_learning_cv` will enter the you created and anything you install with `conda` will be added to this environment. 
* The asterisk * from the `conda env list` output is the currently activated environment. 
* From this point onwards, all commands marked `(deep_learning_cv) $ ` will mean the command was run with the activated environment.

### Installing deep learning libraries
In this short tutorial, to prepare for all situations, we will be installing a wide range of deep learning and computer vision libraries, including:
* OpenCV
* Tensorflow
* Keras
* PyTorch
* MxNet


