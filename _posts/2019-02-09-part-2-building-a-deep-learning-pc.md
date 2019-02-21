---
layout: post
title: Part 2 of 3 - Building a Deep Learning PC
status: completed
---

This is a three part series that I will endeavour to update regularly as I discover better workflows.
1. Part 1: Hardware Drivers and System OS Installation.
2. Part 2 (this post): 
[Development Environment, Frameworks, and IDE Installation](2019-02-09-part-2-building-a-deep-learning-pc.md).
3. Part 3: Configuring Remote Access and Testing your new Development Environment.
4. Part 4 (Optional): Using GPU on Linux without OpenGL and Xorg
5. Part 5 (Optional): Windows Raw Disk Virtual Machine

## Hardware and System OS
Just a reminder these are the hardware and software configurations I have been using as part of this series.

#### Hardware Configurations
```
CPU: Intel Core i9 9900K LGA1151 3.6GHz (5GHz Turbo) 8 Cores, 16 Thread Unlocked 
Motherboard: Asus ROG Maximus XI Extreme*
Memory: 32GB DIMM DDR4 3200MHZ
PSU: Asus ROG Thor 1200W 80+ Platinum
Hard Drive: 2x Samsung 970 Pro 521GB V-NAND NVMe M.2 SSD**
Additional Persistent Storage: 3x Samsung 860 Pro 1TB V-NAND SATA SSD in Raid 0 with mdadm
Onboard GPU: Intel UHD Graphics 630 
Additional GPU: 2x Asus GeForce RTX 2080 Ti Dual with NVIDIA NVLink
```
\* The Asus AI Overclocking feature of this motherboard allows me to run my CPU at a consistent 4.8GHZ without too 
much crazy cooling required. You will need some sort of AIO or custom water cooling to achieve something similar. 

** Windows 10 was installed in one SSD to allow easier configurations of BIOS with utilities provided by Asus.

*** There is still some considerations to use Intel Optane Memory Raid however, there are some concerns with PCI-e 
lanes with adding too many devices in addition to running dual GPU (will update further). 

#### System Configurations
```
Linux Kernel: 4.18.0-14-generic*
OS Version: Ubuntu 18.04.1 LTS Bionic
GPU Driver: nvidia-driver-415*
CUDA Version: CUDA 10.0*
```
\* These are the latest versions in the Ubuntu repository as of this post 2019-02-09.


## Installing Virtual Environment (Anaconda)
When managing software development environments, particularly those with complex libraries such those used for machine 
and deep learning, I prefer to use [Anaconda](https://www.anaconda.com/what-is-anaconda/). You could also use 
[miniconda](https://conda.io/miniconda.html) - a more lightweight version of Anaconda.

```bash
$ cd ~/Downloads
$ wget https://repo.continuum.io/archive/Anaconda3-2018.12-Linux-x86_64.sh
$ chmod +x ./Anaconda3-2018.12-Linux-x86_64.sh
$ ./Anaconda3-2018.12-Linux-x86_64.sh
```

### (Optional) Conda Shell Symlinking

One of the things the Anaconda and Miniconda installer likes to do is create a conda shell as part of your preferred 
shell session. One of the issues I found with this is you are then forced to use the conda version of Python 3 when 
running Python software. This could be an issue if you are required to use Python 2.7 for example which can be 
frustrating to have to write `/usr/bin/python`. 

```bash
$ which python
/home/user/anaconda3/bin/python

$ python -V
Python 3.7.0
```

To avoid this, I prefer not to have the conda PATH set in my 
environment PATH variables and instead symlink the required programs. 

First, remove or comment these lines out of your `.bashrc` file.

```bash
...

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
#__conda_setup="$('/home/user/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
#if [ $? -eq 0 ]; then
#    eval "$__conda_setup"
#else
#    if [ -f "/home/user/anaconda3/etc/profile.d/conda.sh" ]; then
#        . "/home/user/anaconda3/etc/profile.d/conda.sh"
#    else
#        export PATH="/home/user/anaconda3/bin:$PATH"
#    fi
#fi
#unset __conda_setup
# <<< conda initialize <<<
```

Now exit out of the shell session as running `source .bashrc` does not reset the session properly. Then finally, 
symlink the required anaconda3 commands. 

```bash
$ sudo ln -s $HOME/anaconda3/bin/conda /usr/bin/conda
$ sudo ln -s $HOME/anaconda3/bin/activate /usr/bin/activate
$ sudo ln -s $HOME/anaconda3/bin/deactivate /usr/bin/deactivate
$ which python 
/usr/bin/python

$ python -V 
Python 2.7.15rc1
```

### Linking CUDA Toolkit to Conda Environment
Anaconda has recently made the CUDA Toolkit available through the `anaconda` channel. However, the latest version 
posted was CUDA 9.0 and with an NVIDIA Turing Architecture GPU, you will need CUDA 10.0 to work properly. So to while
we will not have the benefit of using our conda virtual environment package manager to manage CUDA, we have already
installed CUDA Toolkit 10.0 as above. To make it work within our env, we will need to do some environment variable
settings. 

```bash
$ source activate deep_learning_cv
(deep_learning_cv) $ conda update mkl
$ conda deactivate
$ mkdir -p $HOME/anaconda3/envs/deep_learning_cv/etc/conda/activate.d
$ mkdir -p $HOME/anaconda3/envs/deep_learning_cv/etc/conda/deactivate.d
$ nano $HOME/anaconda3/envs/deep_learning_cv/etc/conda/activate.d/env_vars.sh
```

##### $HOME/anaconda3/envs/deep_learning_cv/etc/conda/activate.d/env_vars.sh
```bash
#!/bin/sh
export CUDA_HOME="/usr/local/cuda"
export CUPTI="/usr/local/cuda/extras/CUPTI/lib64"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUPTI:$LD_LIBRARY_PATH"
```

```bash
$ nano $HOME/anaconda3/envs/deep_learning_cv/etc/conda/deactivate.d/env_vars.sh
```

##### $HOME/anaconda3/envs/deep_learning_cv/etc/conda/deactivate.d/env_vars.sh

```bash
#!/bin/sh
unset LD_LIBRARY_PATH
```

This will properly set the environment variables when you activate your Conda virtual environment. It will also unset
 them so they do not mess with your local system.

### Creating the deep learning environment

To create an environment with Anaconda, you will generally use the `conda` command in a terminal. To install packages 
for this environment, you will also use either `conda install` or `pip install`.

```bash
$ conda create --name deep_learning_cv python=3.6 numpy pylint -c anaconda
Collecting package metadata: done
Solving environment: done

## Package Plan ##

  environment location: /home/user/anaconda3/envs/deep_learning_cv

  added / updated specs:
    - numpy
    - pylint
    - python=3.6


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    astroid-2.1.0              |           py36_0         271 KB  anaconda
    blas-1.0                   |              mkl           6 KB  anaconda
    isort-4.3.4                |           py36_0          58 KB  anaconda
    lazy-object-proxy-1.3.1    |   py36h14c3975_2          30 KB  anaconda
    mccabe-0.6.1               |           py36_1          14 KB  anaconda
    numpy-1.15.4               |   py36h7e9f1db_0          47 KB  anaconda
    pylint-2.2.2               |           py36_0         828 KB  anaconda
    typed-ast-1.1.0            |   py36h14c3975_0         196 KB  anaconda
    wrapt-1.11.0               |   py36h7b6447c_0          45 KB  anaconda
    ------------------------------------------------------------
                                           Total:         1.5 MB

The following NEW packages will be INSTALLED:
  ...

Proceed ([y]/n)? y


$ source activate deep_learning_cv
$ conda env list
# conda environments:
#
base                     /home/user/anaconda3
deep_learning_cv      *  /home/user/anaconda3/envs/deep_learning_cv

```
Some notes to consider:
* The `base` environment is the conda environment with the default Python (in this case, Python 3.7).
* Using `source activate deep_learning_cv` will enter the you created and anything you install with `conda` will be 
added to this environment. 
* The asterisk `*` from the `conda env list` output is the currently activated environment. 
* From this point onwards, all commands marked `(deep_learning_cv) $ ` will mean the command was run with the activated environment.



### Installing OpenCV from Source

[OpenCV (Open Source Computer Vision)](https://opencv.org/) is a library that is very popular these days when doing Deep Learning with 
Computer Vision or any image processing. It has a C++ and Python interface that we can make use of during development. 

While there are ways to install OpenCV using the Anaconda channel, if we prefer to have better control over the 
library (meaning we can install both the Python and C++ interface), then the best way is to compile it from source, 
installing it within our local environment, and then soft linking it from within the Anaconda environment.

Firstly, ensure you install these dependencies:

```bash
$ sudo apt install build-essential cmake unzip pkg-config pylint libjpeg-dev \ 
libpng-dev libtiff-dev libavcodec-dev libavformat-dev libdc1394-22-dev \
libx264-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libeigen3-dev \ 
gfortran libhdf5-serial-dev python3-dev python3-tk python-imaging-tk \
ubuntu-restricted-extras libgtk-3-dev libatlas-base-dev libgstreamer1.0-dev \
libgstreamer-plugins-base1.0-dev libavresample-dev libgflags2.2 libgflags-dev
```

Another dependency required by OpenCV that is not available in the `apt` repository with Ubuntu 18.04 is `libjasper`.
 To install, follow these manual steps:

```bash
$ wget http://security.ubuntu.com/ubuntu/pool/main/j/jasper/libjasper-dev_1.900.1-debian1-2.4ubuntu1.2_amd64.deb
$ wget http://security.ubuntu.com/ubuntu/pool/main/j/jasper/libjasper1_1.900.1-debian1-2.4ubuntu1.2_amd64.deb
$ sudo apt-get install ./libjasper1_1.900.1-debian1-2.4ubuntu1.2_amd64.deb \ 
  ./libjasper-dev_1.900.1-debian1-2.4ubuntu1.2_amd64.deb
```

Lets prepare our installation directory:

```bash
$ mkdir opencv_install && cd opencv_install
$ git clone https://github.com/opencv/opencv.git
$ git clone https://github.com/opencv/opencv_contrib.git
$ cd opencv; git checkout 3.4.5
$ cd ../opencv_contrib; git checkout 3.4.5
```

An issue noted in the following [#9953](https://github.com/opencv/opencv/issues/9953) and then solved in 
[#12957](https://github.com/opencv/opencv/issues/12957) to get OpenBLAS to work in Ubuntu 18.04. 

Install the required dependencies:
```bash
$ sudo apt install libopenblas-base libopenblas-dev liblapacke-dev
$ sudo ln -s /usr/include/lapacke.h /usr/include/x86_64-linux-gnu # corrected path for the library
$ nano ../opencv/cmake/OpenCVFindOpenBLAS.cmake
```

From there, fix the following two lines:

```
SET(Open_BLAS_INCLUDE_SEARCH_PATHS
  $ENV{OpenBLAS_HOME}
  $ENV{OpenBLAS_HOME}/include
  /opt/OpenBLAS/include
  /usr/local/include/openblas
  /usr/include/openblas
  /usr/local/include/openblas-base
  /usr/include/openblas-base
  /usr/local/include
  /usr/include
)

SET(Open_BLAS_LIB_SEARCH_PATHS
        $ENV{OpenBLAS}cd
        $ENV{OpenBLAS}/lib
        $ENV{OpenBLAS_HOME}
        $ENV{OpenBLAS_HOME}/lib
        /opt/OpenBLAS/lib
        /usr/local/lib64
        /usr/local/lib
        /lib/openblas-base
        /lib64/
        /lib/
        /usr/lib/openblas-base
        /usr/lib64
        /usr/lib
)
```

to become...

```bash
SET(Open_BLAS_INCLUDE_SEARCH_PATHS
  /usr/include/x86_64-linux-gnu
)

SET(Open_BLAS_LIB_SEARCH_PATHS
  /usr/lib/x86_64-linux-gnu
)
```

Finally, we can download OpenCV from their Github repository and compile the releases we want. In this case, we are 
going to install version `3.4.5`.

```bash
$ cd .. && nano xcmake.sh 
```

##### xcmake.sh 
```bash
#!/bin/bash

CONDA_ENV_PATH=/home/user/anaconda3/envs
CONDA_ENV_NAME=deep_learning_cv
WHERE_OPENCV=../opencv
WHERE_OPENCV_CONTRIB=../opencv_contrib

cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D BUILD_OPENCV_PYTHON3=ON \
      -D INSTALL_C_EXAMPLES=ON \
      -D INSTALL_PYTHON_EXAMPLES=ON \
      -D PYTHON3_EXECUTABLE=$CONDA_ENV_PATH/$CONDA_ENV_NAME/bin/python \
      -D WITH_STREAMER=ON \
      -D WITH_CUDA=ON \
      -D BUILD_opencv_cudacodec=OFF \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D OPENCV_EXTRA_MODULES_PATH=$WHERE_OPENCV_CONTRIB/modules \
      -D BUILD_EXAMPLES=ON $WHERE_OPENCV
```

```bash
$ sudo chmod u+x xcmake.sh
$ mkdir build && cd build
$ ../xcmake.sh
```

If all dependencies are met and everything goes well, you should see the following output at the end:
```bash
...

--   Other third-party libraries:
--     Intel IPP:                   2019.0.0 Gold [2019.0.0]
--            at:                   /home/user/opencv_install/build/3rdparty/ippicv/ippicv_lnx/icv
--     Intel IPP IW:                sources (2019.0.0)
--               at:                /home/user/opencv_install/build/3rdparty/ippicv/ippicv_lnx/iw
--     Lapack:                      YES (/usr/lib/x86_64-linux-gnu/libopenblas.so)
--     Eigen:                       YES (ver 3.3.4)
--     Custom HAL:                  NO
--     Protobuf:                    build (3.5.1)
-- 
--   NVIDIA CUDA:                   YES (ver 10.0, CUFFT CUBLAS NVCUVID)
--     NVIDIA GPU arch:             30 35 37 50 52 60 61 70 75
--     NVIDIA PTX archs:
-- 
--   OpenCL:                        YES (no extra features)
--     Include path:                /home/user/opencv_install/opencv/3rdparty/include/opencl/1.2
--     Link libraries:              Dynamic load
-- 
--   Python 2:
--     Interpreter:                 /usr/bin/python2.7 (ver 2.7.15)
--     Libraries:                   /usr/lib/x86_64-linux-gnu/libpython2.7.so (ver 2.7.15rc1)
--     numpy:                       /usr/lib/python2.7/dist-packages/numpy/core/include (ver 1.13.3)
--     install path:                lib/python2.7/dist-packages/cv2/python-2.7
-- 
--   Python 3:
--     Interpreter:                 /home/user/anaconda3/envs/deep_learning_cv/bin/python (ver 3.6.8)
--     Libraries:                   /home/user/anaconda3/envs/deep_learning_cv/lib/libpython3.6m.so (ver 3.6.8)
--     numpy:                       /home/user/anaconda3/envs/deep_learning_cv/lib/python3.6/site-packages/numpy/core/include (ver 1.15.4)
--     install path:                lib/python3.6/site-packages/cv2/python-3.6
-- 
--   Python (for build):            /usr/bin/python2.7
--     Pylint:                      /home/user/anaconda3/envs/deep_learning_cv/bin/pylint (ver: 3.6.8, checks: 163)
-- 
--   Java:                          
--     ant:                         NO
--     JNI:                         /usr/lib/jvm/default-java/include /usr/lib/jvm/default-java/include/linux /usr/lib/jvm/default-java/include
--     Java wrappers:               NO
--     Java tests:                  NO
-- 
--   Install to:                    /usr/local
-- -----------------------------------------------------------------
-- 
-- Configuring done
-- Generating done
-- Build files have been written to: /home/user/opencv_install/build
```

Time to compile and install:

```bash
$ make -j4
$ sudo make install
$ sudo ldconfig
```

Check that you have installed OpenCV:

```bash
$ pkg-config --modversion opencv
3.4.5
```

Finally, symlinking to the conda virtual environment we created earlier:
```bash
$ sudo ln -s /usr/local/lib/python3.6/site-packages/cv2/python-3.6/cv2.cpython-36m-x86_64-linux-gnu.so \
  /home/user/anaconda3/envs/deep_learning_cv/lib/python3.6/site-packages/cv2.so
$ source activate deep_learning_cv
(deep_learning_cv) $ python -c "import cv2; print(cv2.__version__)"
3.4.5
```

### Installing mxnet
Ubuntu 18.04 comes shipped with `gcc` and `g++` version 7. However, to compile MxNet from source, you will need to 
use `gcc` version 6. We can easily change the default `gcc` used by doing adding `gcc-6` and `g++-6` as an alternative in the configuration files. Make the following script and then run it.

##### configure_multiple_cc.sh 
 ```bash
#!/bin/bash

sudo apt update && sudo apt install gcc-7 g++-7 gcc-6 g++-6

sudo update-alternative --remove-all gcc
sudo update-alternative --remove-all g++

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 100
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 90

sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-6 90

sudo update-alternatives --set cc /usr/bin/gcc
sudo update-alternatives --set c++ /usr/bin/g++
 ```

Then make it executable and then run it and select the appropriate `gcc` and `g++` to use for now.

```bash
$ sudo chmod +x config_multiple_cc.sh
$ sudo update-alternatives --config gcc
There are 2 choices for the alternative gcc (providing /usr/bin/gcc).

  Selection    Path            Priority   Status
------------------------------------------------------------
* 0            /usr/bin/gcc-7   100       auto mode
  1            /usr/bin/gcc-6   90        manual mode
  2            /usr/bin/gcc-7   100       manual mode

Press <enter> to keep the current choice[*], or type selection number: 1
update-alternatives: using /usr/bin/gcc-6 to provide /usr/bin/gcc (gcc) in manual mode

$ gcc --version
gcc (Ubuntu 6.5.0-2ubuntu1~18.04) 6.5.0 20181026
...

$ sudo update-alternatives --config g++
There are 2 choices for the alternative g++ (providing /usr/bin/g++).

  Selection    Path            Priority   Status
------------------------------------------------------------
* 0            /usr/bin/g++-7   100       auto mode
  1            /usr/bin/g++-6   90        manual mode
  2            /usr/bin/g++-7   100       manual mode

Press <enter> to keep the current choice[*], or type selection number: 1
update-alternatives: using /usr/bin/g++-6 to provide /usr/bin/g++ (g++) in manual mode

$ g++ --version
g++ (Ubuntu 6.5.0-2ubuntu1~18.04) 6.5.0 20181026
...

```

We can now clone and install mxnet:

```bash
(deep_learning_cv) $ git clone --recursive --no-checkout https://github.com/apache/incubator-mxnet.git mxnet
(deep_learning_cv) $ mv mxnet /home/user/.local/share/
(deep_learning_cv) $ git checkout 1.3.1
(deep_learning_cv) $ git submodule update --init
(deep_learning_cv) $ make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 \
USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1 \
USE_LAPACK_PATH=/usr/lib/x86_64-linux-gnu USE_LAPACK=1 USE_MKLDNN=1
```

Once that is complete, we simply link it to our Anaconda3 virtual environment and then test:

```bash
$ ln -s $HOME/.local/share/mxnet $HOME/anaconda3/envs/deep_learning_cv/lib/python3.6/site-packages/mxnet
$ source activate deep_learning_cv
(deep_learning_cv) $ python

Python 3.6.8 |Anaconda, Inc.| (default, Dec 30 2018, 01:22:34) 
[GCC 7.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import mxnet
>>>

```

### Installing Tensorflow GPU from Source

So as I was writing this post, I came across some issues using CUDA Toolkit version 10.0 with the `tensorflow-gpu` package as part of the PyPi and Conda repositories. Using the `anaconda` channel would have been, however, it installs CUDA Toolkit 9.0 and cuDNN 7 by default. Since we already have CUDA 10.0 installed, I though it would be more prudent in this case to install `tensorflow` from source files so that it works with CUDA 10.0 and cuDNN 7.3.1. There are very few tutorials out there that specifically show you how to manage this using an Anaconda virtual environment. However, there are some clues [here](https://www.pytorials.com/how-to-install-tensorflow-gpu-with-cuda-10-0-for-python-on-ubuntu/) and [here](https://medium.com/@vitali.usau/install-cuda-10-0-cudnn-7-3-and-build-tensorflow-gpu-from-source-on-ubuntu-18-04-3daf720b83fe). The official [Tensorflow installation guide](https://www.tensorflow.org/install/source) also has steps for virtual environments. 

#### NCCL 

The NVIDIA Collective Communications Library (NCCL) allows you to handle communication across multiple-GPU simultaneously. 

As of this post, the latest version is `2.4.2, for CUDA 10.0, Jan 29. 2019`. Go to this [link](https://developer.nvidia.com/nccl/nccl-download) and click on _Local installer for Ubuntu18.04_ to your `$HOME/Downloads$` directory.

```bash
$ sudo dpkg -i $HOME/Downloads/nccl-repo-ubuntu1804-2.4.2-ga-cuda10.0_1-1_amd64.deb
$ sudo apt-key add /var/nccl-repo-2.4.2-ga-cuda10.0/7fa2af80.pub
$ sudo apt update
$ sudo apt install libnccl2 libnccl-dev
$ sudo ldconfig 
```

#### Dependencies

```bash
$ source activate deep_learning_cv
$ pip install -U pip six wheel mock
$ pip install -U keras_applications==1.0.6 --no-deps
$ pip install -U keras_preprocessing==1.0.5 --no-deps
```

#### Bazel

Bazel is a build tool used for building Tensorflow.

```bash
$ wget https://github.com/bazelbuild/bazel/releases/download/0.21.0/bazel-0.21.0-installer-linux-x86_64.sh
$ sudo chmod +x bazel-0.21.0-installer-linux-x86_64.sh
./bazel-0.21.0-installer-linux-x86_64.sh
$ echo 'export PATH="$PATH:$HOME/bin"' >> ~/.bashrc
$ source $HOME/.bashrc
$ sudo ldconfig
$ bazel version

WARNING: --batch mode is deprecated. Please instead explicitly shut down your Bazel server using the command "bazel shutdown".
INFO: Invocation ID: 60babe73-16ad-42b7-8474-9f617853c4e2
Build label: 0.21.0
Build target: bazel-out/k8-opt/bin/src/main/java/com/google/devtools/build/lib/bazel/BazelServer_deploy.jar
Build time: Wed Dec 19 12:58:44 2018 (1545224324)
Build timestamp: 1545224324
Build timestamp as int: 1545224324
```

#### (Optional) Install TensorRT

TensorRT is a C++ library that has incredibly good performance for NVIDIA GPUs. As part of the project I am going to be working on soon, I will need to implement a Deep Learning algorithm in TensorRT to optimize the performance. So due to this, I am going to install TensorRT as well before installing Tensorflow so I can have both libraries configured properly. The steps to install TensorRT can be found [here](https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html).

First, [download](https://developer.nvidia.com/tensorrt) the appropriate version after you agree to the terms and conditions. In this case, we will be using _TensorRT 5.0.2.6 GA for Ubuntu 1804 and CUDA 10.0 DEB local repo packages_.  This version is currently supported by the Tensorflow version we will be installing below.

```bash
$ sudo dpkg -i nv-tensorrt-repo-ubuntu1804-cuda10.0-trt5.0.2.6-ga-20181009_1-1_amd64.deb
$ sudo apt-key add /var/nv-tensorrt-repo-cuda10.0-trt5.0.2.6-ga-20181009/7fa2af80.pub
OK

$ sudo apt install libnvinfer5 libnvinfer-dev libnvinfer-samples tensorrt
$ sudo apt install python3-libnvinfer-dev
$ sudo apt install uff-converter-tf
```

#### Tensorflow GPU

Now we download the latest version of Tensorflow and begin the process of installing it with proper variables. Please be very careful which directories you select to make this work. 

```bash
$ cd $HOME
$ git clone https://github.com/tensorflow/tensorflow.git
$ cd tensorflow
$ git checkout r1.13
$ ./configure
```

Now we begin changing the build configuration variables as appropriate.

```bash
WARNING: --batch mode is deprecated. Please instead explicitly shut down your Bazel server using the command "bazel shutdown".
INFO: Invocation ID: b6fc6907-7a4d-47d6-addf-25915920fc92
You have bazel 0.21.0 installed.
Please specify the location of python. [Default is /home/codeninja/anaconda3/envs/deep_learning_cv/bin/python]: /home/codeninja/anaconda3/envs/deep_learning_cv/bin/python


Found possible Python library paths:
  /home/codeninja/anaconda3/envs/deep_learning_cv/lib/python3.6/site-packages
Please input the desired Python library path to use.  Default is [/home/codeninja/anaconda3/envs/deep_learning_cv/lib/python3.6/site-packages]
/home/codeninja/anaconda3/envs/deep_learning_cv/lib/python3.6/site-packages
Do you wish to build TensorFlow with XLA JIT support? [Y/n]: Y
XLA JIT support will be enabled for TensorFlow.

Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: N
No OpenCL SYCL support will be enabled for TensorFlow.

Do you wish to build TensorFlow with ROCm support? [y/N]: N
No ROCm support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: Y
CUDA support will be enabled for TensorFlow.

Please specify the CUDA SDK version you want to use. [Leave empty to default to CUDA 10.0]: 10.0


Please specify the location where CUDA 10.0 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]: /usr/local/cuda


Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 7]: 7.4.2


Please specify the location where cuDNN 7 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]: /usr/local/cuda


Do you wish to build TensorFlow with TensorRT support? [y/N]: Y
TensorRT support will be enabled for TensorFlow.

Please specify the location where TensorRT is installed. [Default is /usr/lib/x86_64-linux-gnu]:


Please specify the locally installed NCCL version you want to use. [Default is to use https://github.com/nvidia/nccl]: 2.4.2


NCCL libraries found in /usr/lib/x86_64-linux-gnu/libnccl.so
This looks like a system path.
Assuming NCCL header path is /usr/include
Please specify a list of comma-separated Cuda compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
Please note that each additional compute capability significantly increases your build time and binary size. [Default is: 7.5]: 7.5


Do you want to use clang as CUDA compiler? [y/N]: N
nvcc will be used as CUDA compiler.

Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]: /usr/bin/gcc


Do you wish to build TensorFlow with MPI support? [y/N]: N
No MPI support will be enabled for TensorFlow.

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native -Wno-sign-compare]: -march=native


Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: N
Not configuring the WORKSPACE for Android builds.

Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See .bazelrc for more details.
	--config=mkl         	# Build with MKL support.
	--config=monolithic  	# Config for mostly static monolithic build.
	--config=gdr         	# Build with GDR support.
	--config=verbs       	# Build with libverbs support.
	--config=ngraph      	# Build with Intel nGraph support.
	--config=dynamic_kernels	# (Experimental) Build kernels into separate shared objects.
Preconfigured Bazel build configs to DISABLE default on features:
	--config=noaws       	# Disable AWS S3 filesystem support.
	--config=nogcp       	# Disable GCP support.
	--config=nohdfs      	# Disable HDFS support.
	--config=noignite    	# Disable Apacha Ignite support.
	--config=nokafka     	# Disable Apache Kafka support.
	--config=nonccl      	# Disable NVIDIA NCCL support.
Configuration finished
```

#### Building with bazel

The next step is to build from the configurations we just completed. Please note this may take a long time depending on the computation power of your CPU and system. 

The following will build a `pip` package that we can use in our virtual environment later. 

```bash
$ bazel build --config=opt --config=mkl --config=cuda //tensorflow/tools/pip_package:build_pip_package

Target //tensorflow/tools/pip_package:build_pip_package up-to-date:
  bazel-bin/tensorflow/tools/pip_package/build_pip_package
INFO: Elapsed time: 3629.167s, Critical Path: 286.01s
INFO: 15303 processes: 15303 local.
INFO: Build completed successfully, 18989 total actions
```

**Some Notes:**

* add "--config=mkl" if you want Intel MKL support for newer intel cpu for faster training on cpu
* add "--config=monolithic" if you want static monolithic build (try this if build failed)
* add "--local_resources 2048,.5,1.0" if your PC has low ram causing Segmentation fault or other related errors

#### (Optional) Build Tensorflow with Intel MKL with AVX, AVX2, and AVX512

```bash
$ bazel build --config=opt --config=cuda --config=mkl -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mavx512f --copt=-mavx512pf --copt=-mavx512cd --copt=-mavx512er //tensorflow/tools/pip_package:build_pip_package
```



If everything worked fine, it should look like the output above. Finally, we need to make the `pip` package and then install it to our Anaconda virtual env. 

```bash
$ bazel-bin/tensorflow/tools/pip_package/build_pip_package tensorflow_pkg
$ $ cd tensorflow_pkg
$ source activate deep_learning_cv
(deep_learning_cv) $ pip install tensorflow-1.13.0rc1-cp36-cp36m-linux_x86_64.whl

```

To finish off, we will just quickly test it.

```bash
(deep_learning_cv) $ python
Python 3.6.8 |Anaconda, Inc.| (default, Dec 30 2018, 01:22:34) 
[GCC 7.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()

2019-02-10 20:43:10.467012: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 3600000000 Hz
2019-02-10 20:43:10.467622: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x55c2348570d0 executing computations on platform Host. Devices:
2019-02-10 20:43:10.467637: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): <undefined>, <undefined>
2019-02-10 20:43:10.630042: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:998] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2019-02-10 20:43:10.630525: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x55c2346448e0 executing computations on platform CUDA. Devices:
2019-02-10 20:43:10.630535: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): GeForce RTX 2080 Ti, Compute Capability 7.5
2019-02-10 20:43:10.630820: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1433] Found device 0 with properties: 
name: GeForce RTX 2080 Ti major: 7 minor: 5 memoryClockRate(GHz): 1.56
pciBusID: 0000:01:00.0
totalMemory: 10.73GiB freeMemory: 9.20GiB
2019-02-10 20:43:10.630830: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0
2019-02-10 20:43:10.858953: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-02-10 20:43:10.858978: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0 
2019-02-10 20:43:10.858982: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N 
2019-02-10 20:43:10.859211: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 8861 MB memory) -> physical GPU (device: 0, name: GeForce RTX 2080 Ti, pci bus id: 0000:01:00.0, compute capability: 7.5)
2019-02-10 20:43:10.860079: I tensorflow/core/common_runtime/process_util.cc:71] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.

>>> print(sess.run(hello))

b'Hello, TensorFlow!'

>>> 
```

If your output is the same as above, then you have successfully installed TensorFlow with GPU.

### Installing PyTorch

To install PyTorch from source, ensure you create the `LD_LIBRARY_PATH` environment variables for the Anaconda virtual environment as we did previously. Apart from that, we can now install some dependencies:

```bash
$ source activate deep_learning_cv
(deep_learning_cv) $ conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing
(deep_learning_cv) $ pip install pytorch
(deep_learning_cv) $ conda install -c cpbotha magma-cuda10
```

Get the PyTorch source and install with some custom variables.

```bash
(deep_learning_cv) $ git clone --recursive https://github.com/pytorch/pytorch
(deep_learning_cv) $ cd pytorch
(deep_learning_cv) $ export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
(deep_learning_cv) $ sudo apt install protobuf-compiler libprotobuf-dev
(deep_learning_cv) $ export USE_OPENCV=ON
(deep_learning_cv) $ export BLAS=OpenBLAS
(deep_learning_cv) $ export USE_TENSORRT=ON
(deep_learning_cv) $ export MAX_JOBS=16
(deep_learning_cv) $ export CUDNN_LIB_DIR="$CUDA_HOME/lib64/"
(deep_learning_cv) $ export CUDNN_INCLUDE="$CUDA_HOME/include/" 
(deep_learning_cv) $ python setup.py install
```

Now let’s just run a quick test:

```bash
(deep_learning_cv) $ cd ..
(deep_learning_cv) $ python -c 'import torch' 2>/dev/null && echo "Success" || echo "Failure"
Success

(deep_learning_cv) $ python -c 'import torch; print(torch.cuda.is_available())'
True

```

If you want to do a more comprehensive test, do the following:

```bash
(deep_learning_cv) $ cd $HOME/pytorch
(deep_learning_cv) $ python test/run_test.py --exclude cpp_extensions

Test executor: ['/home/user/anaconda3/envs/deep_learning_cv/bin/python']
Running test_autograd ... [2019-02-11 01:28:46.842847]
................................................................................
```

Finally, we will need to install `torchvision`:

```bash
(deep_learning_cv) $ cd $HOME/pytorch
(deep_learning_cv) $ git clone https://github.com/pytorch/vision
(deep_learning_cv) $ cd vision
(deep_learning_cv) $ python setup.py install
```



### Installing other deep learning libraries

Finally, there is one other library we are going to install and test with GPU. That library is Keras.

```bash
(deep_learning_cv) $ pip install keras
```

To test that it works with our TensorFlow GPU backend, we will run the following:

```bash
(deep_learning_cv) $ python

Python 3.6.8 |Anaconda, Inc.| (default, Dec 30 2018, 01:22:34) 
[GCC 7.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from keras import backend as K
>>> K.tensorflow_backend._get_available_gpus()
2019-02-11 01:44:00.385863: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 3600000000 Hz
2019-02-11 01:44:00.387428: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x558c0ba50990 executing computations on platform Host. Devices:
2019-02-11 01:44:00.387485: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): <undefined>, <undefined>
2019-02-11 01:44:00.671487: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:998] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2019-02-11 01:44:00.672758: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x558c0b1a6b10 executing computations on platform CUDA. Devices:
2019-02-11 01:44:00.672800: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): GeForce RTX 2080 Ti, Compute Capability 7.5
2019-02-11 01:44:00.673333: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1433] Found device 0 with properties: 
name: GeForce RTX 2080 Ti major: 7 minor: 5 memoryClockRate(GHz): 1.56
pciBusID: 0000:01:00.0
totalMemory: 10.73GiB freeMemory: 9.07GiB
2019-02-11 01:44:00.673355: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0
2019-02-11 01:44:01.180883: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-02-11 01:44:01.180939: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0 
2019-02-11 01:44:01.180954: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N 
2019-02-11 01:44:01.181396: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 8733 MB memory) -> physical GPU (device: 0, name: GeForce RTX 2080 Ti, pci bus id: 0000:01:00.0, compute capability: 7.5)
2019-02-11 01:44:01.185912: I tensorflow/core/common_runtime/process_util.cc:71] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
['/job:localhost/replica:0/task:0/device:GPU:0']
>>> 
```

### That is all folks! 

This was an extremely long post but there were a lot of frameworks and libraries installed. If you only needed some of them, you could have easily skipped the others. Thanks for reading. I will endeavour to get another post out soon. 