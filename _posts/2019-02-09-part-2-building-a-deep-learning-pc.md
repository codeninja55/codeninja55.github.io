---
layout: post
title: Part 2 of 3 - Building a Deep Learning PC
status: draft
---

This is a three part series that I will endeavour to update regularly as I discover better workflows.
1. Part 1: Hardware Drivers and System OS Installation.
2. Part 2 (this post): [Development Environment, Frameworks, and IDE Installation](2018-12-28-part-2-building-a-deep-learning-pc.md).
3. Part 3: Configuring Remote Access and Testing your new Development Environment.
4. Part 4 (Optional): Using GPU on Linux without OpenGL and Xorg
5. Part 5 (Optional): Windows Raw Disk Virtual Machine

## Hardware and System OS Installation
Just a reminder these are the hardware and software configurations I have been using as part of this series.

#### Hardware Configurations
```
CPU: Intel Core i9 9900K LGA1151 3.6GHz (5GHz Turbo) 8 Cores, 16 Thread Unlocked 
Motherboard: Asus ROG Maximus XI Extreme*
Memory: 32GB DIMM DDR4 3200MHZ
PSU: Asus ROG Thor 1200W 80+ Platinum
Hard Drive: 2x Samsung 970 Pro 521GB V-NAND NVMe M.2 SSD**
Additional Persistent Storage: 6x Samsung 860 Pro 1TB V-NAND SATA SSD in Raid 0 with Intel Rapid Storage***
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
and deep learning, I prefer to use [Anaconda](https://www.anaconda.com/what-is-anaconda/). You could also use [miniconda](https://conda.io/miniconda.html) - a more lightweight version of Anaconda.

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

  astroid            anaconda/linux-64::astroid-2.1.0-py36_0
  blas               anaconda/linux-64::blas-1.0-mkl
  ca-certificates    anaconda/linux-64::ca-certificates-2019.1.23-0
  certifi            anaconda/linux-64::certifi-2018.11.29-py36_0
  intel-openmp       anaconda/linux-64::intel-openmp-2019.1-144
  isort              anaconda/linux-64::isort-4.3.4-py36_0
  lazy-object-proxy  anaconda/linux-64::lazy-object-proxy-1.3.1-py36h14c3975_2
  libedit            anaconda/linux-64::libedit-3.1.20181209-hc058e9b_0
  libffi             anaconda/linux-64::libffi-3.2.1-h4deb6c0_3
  libgcc-ng          anaconda/linux-64::libgcc-ng-8.2.0-hdf63c60_1
  libgfortran-ng     anaconda/linux-64::libgfortran-ng-7.3.0-hdf63c60_0
  libstdcxx-ng       anaconda/linux-64::libstdcxx-ng-8.2.0-hdf63c60_1
  mccabe             anaconda/linux-64::mccabe-0.6.1-py36_1
  mkl                anaconda/linux-64::mkl-2019.1-144
  mkl_fft            anaconda/linux-64::mkl_fft-1.0.10-py36ha843d7b_0
  mkl_random         anaconda/linux-64::mkl_random-1.0.2-py36hd81dba3_0
  ncurses            anaconda/linux-64::ncurses-6.1-he6710b0_1
  numpy              anaconda/linux-64::numpy-1.15.4-py36h7e9f1db_0
  numpy-base         anaconda/linux-64::numpy-base-1.15.4-py36hde5b4d6_0
  openssl            anaconda/linux-64::openssl-1.1.1-h7b6447c_0
  pip                anaconda/linux-64::pip-19.0.1-py36_0
  pylint             anaconda/linux-64::pylint-2.2.2-py36_0
  python             anaconda/linux-64::python-3.6.8-h0371630_0
  readline           anaconda/linux-64::readline-7.0-h7b6447c_5
  setuptools         anaconda/linux-64::setuptools-40.7.3-py36_0
  six                anaconda/linux-64::six-1.12.0-py36_0
  sqlite             anaconda/linux-64::sqlite-3.26.0-h7b6447c_0
  tk                 anaconda/linux-64::tk-8.6.8-hbc83047_0
  typed-ast          anaconda/linux-64::typed-ast-1.1.0-py36h14c3975_0
  wheel              anaconda/linux-64::wheel-0.32.3-py36_0
  wrapt              anaconda/linux-64::wrapt-1.11.0-py36h7b6447c_0
  xz                 anaconda/linux-64::xz-5.2.4-h14c3975_4
  zlib               anaconda/linux-64::zlib-1.2.11-h7b6447c_3


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
$ sudo apt install build-essential cmake unzip pkg-config pylint libjpeg-dev libpng-dev libtiff-dev \
  libavcodec-dev libavformat-dev libdc1394-22-dev libx264-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev \
  libeigen3-dev gfortran libhdf5-serial-dev python3-dev python3-tk python-imaging-tk ubuntu-restricted-extras \ 
  libgtk-3-dev libatlas-base-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libavresample-dev 
```

Another dependency required by OpenCV that is not available in the `apt` repository with Ubuntu 18.04 is `libjasper`.
 To install, follow these manual steps:
```bash
$ wget http://security.ubuntu.com/ubuntu/pool/main/j/jasper/libjasper-dev_1.900.1-debian1-2.4ubuntu1.2_amd64.deb
$ wget http://security.ubuntu.com/ubuntu/pool/main/j/jasper/libjasper1_1.900.1-debian1-2.4ubuntu1.2_amd64.deb
$ sudo apt-get install ./libjasper1_1.900.1-debian1-2.4ubuntu1.2_amd64.deb \ 
  ./libjasper-dev_1.900.1-debian1-2.4ubuntu1.2_amd64.deb
```

Finally, another issue noted in the following [#9953](https://github.com/opencv/opencv/issues/9953) and then solved in 
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
going to install version `4.0.1`, however, other authors of tutorials recommend `3.4.5`.

```bash
$ mkdir opencv_install && cd opencv_install
$ git clone https://github.com/opencv/opencv.git
$ git clone https://github.com/opencv/opencv_contrib.git
$ cd opencv; git checkout 4.0.1
$ cd ../opencv_contrib; git checkout 4.0.1
$ cd .. && nano xcmake.sh 
```

##### xcmake.sh 
```bash
#!/bin/bash
WHERE_OPENCV=../opencv
WHERE_OPENCV_CONTRIB=../opencv_contrib
PYTHON3_HOME=/usr/bin/python3

cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D INSTALL_C_EXAMPLES=ON \
      -D INSTALL_PYTHON_EXAMPLES=ON \
      -D BUILD_OPENCV_PYTHON3=ON \
      -D WITH_STREAMER=ON \
      -D WITH_CUDA=ON \
      -D WITH_NVCUVID=ON \
      -D BUILD_opencv_cudacodec=OFF \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D OPENCV_EXTRA_MODULES_PATH=$WHERE_OPENCV_CONTRIB/modules \
      -D PYTHON3_EXECUTABLE=$PYTHON3_HOME \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D BUILD_EXAMPLES=ON $WHERE_OPENCV
```

```bash
$ sudo chmod u+x xcmake.sh
$ mkdir build && cd build
$ ../xcmake.sh
```

If all dependencies are met and everything goes well, you should see the following output at the end:
```bash
-- General configuration for OpenCV 4.0.1 =====================================
--   Version control:               4.0.1-dirty
-- 
--   Extra modules:
--     Location (extra):            /home/user/opencv_install/opencv_contrib/modules
--     Version control (extra):     4.0.1
-- 
--   Platform:
--     Timestamp:                   2019-02-09T12:22:38Z
--     Host:                        Linux 4.18.0-15-lowlatency x86_64
--     CMake:                       3.10.2
--     CMake generator:             Unix Makefiles
--     CMake build tool:            /usr/bin/make
--     Configuration:               RELEASE
-- 
--   CPU/HW features:
--     Baseline:                    SSE SSE2 SSE3
--       requested:                 SSE3
--     Dispatched code generation:  SSE4_1 SSE4_2 FP16 AVX AVX2 AVX512_SKX
--       requested:                 SSE4_1 SSE4_2 AVX FP16 AVX2 AVX512_SKX
--       SSE4_1 (7 files):          + SSSE3 SSE4_1
--       SSE4_2 (2 files):          + SSSE3 SSE4_1 POPCNT SSE4_2
--       FP16 (1 files):            + SSSE3 SSE4_1 POPCNT SSE4_2 FP16 AVX
--       AVX (5 files):             + SSSE3 SSE4_1 POPCNT SSE4_2 AVX
--       AVX2 (13 files):           + SSSE3 SSE4_1 POPCNT SSE4_2 FP16 FMA3 AVX AVX2
--       AVX512_SKX (1 files):      + SSSE3 SSE4_1 POPCNT SSE4_2 FP16 FMA3 AVX AVX2 AVX_512F AVX512_SKX
-- 
--   C/C++:
--     Built as dynamic libs?:      YES
--     C++ Compiler:                /usr/bin/c++  (ver 7.3.0)
--     C++ flags (Release):         -fsigned-char -W -Wall -Werror=return-type -Werror=non-virtual-dtor -Werror=address -Werror=sequence-point -Wformat -Werror=format-security -Wmissing-declarations -Wundef -Winit-self -Wpointer-arith -Wshadow -Wsign-promo -Wuninitialized -Winit-self -Wno-narrowing -Wno-delete-non-virtual-dtor -Wno-comment -Wimplicit-fallthrough=3 -Wno-strict-overflow -fdiagnostics-show-option -Wno-long-long -pthread -fomit-frame-pointer -ffast-math -ffunction-sections -fdata-sections  -msse -msse2 -msse3 -fvisibility=hidden -fvisibility-inlines-hidden -O3 -DNDEBUG  -DNDEBUG
--     C++ flags (Debug):           -fsigned-char -W -Wall -Werror=return-type -Werror=non-virtual-dtor -Werror=address -Werror=sequence-point -Wformat -Werror=format-security -Wmissing-declarations -Wundef -Winit-self -Wpointer-arith -Wshadow -Wsign-promo -Wuninitialized -Winit-self -Wno-narrowing -Wno-delete-non-virtual-dtor -Wno-comment -Wimplicit-fallthrough=3 -Wno-strict-overflow -fdiagnostics-show-option -Wno-long-long -pthread -fomit-frame-pointer -ffast-math -ffunction-sections -fdata-sections  -msse -msse2 -msse3 -fvisibility=hidden -fvisibility-inlines-hidden -g  -O0 -DDEBUG -D_DEBUG
--     C Compiler:                  /usr/bin/cc
--     C flags (Release):           -fsigned-char -W -Wall -Werror=return-type -Werror=non-virtual-dtor -Werror=address -Werror=sequence-point -Wformat -Werror=format-security -Wmissing-declarations -Wmissing-prototypes -Wstrict-prototypes -Wundef -Winit-self -Wpointer-arith -Wshadow -Wuninitialized -Winit-self -Wno-narrowing -Wno-comment -Wimplicit-fallthrough=3 -Wno-strict-overflow -fdiagnostics-show-option -Wno-long-long -pthread -fomit-frame-pointer -ffast-math -ffunction-sections -fdata-sections  -msse -msse2 -msse3 -fvisibility=hidden -O3 -DNDEBUG  -DNDEBUG
--     C flags (Debug):             -fsigned-char -W -Wall -Werror=return-type -Werror=non-virtual-dtor -Werror=address -Werror=sequence-point -Wformat -Werror=format-security -Wmissing-declarations -Wmissing-prototypes -Wstrict-prototypes -Wundef -Winit-self -Wpointer-arith -Wshadow -Wuninitialized -Winit-self -Wno-narrowing -Wno-comment -Wimplicit-fallthrough=3 -Wno-strict-overflow -fdiagnostics-show-option -Wno-long-long -pthread -fomit-frame-pointer -ffast-math -ffunction-sections -fdata-sections  -msse -msse2 -msse3 -fvisibility=hidden -g  -O0 -DDEBUG -D_DEBUG
--     Linker flags (Release):      
--     Linker flags (Debug):        
--     ccache:                      NO
--     Precompiled headers:         YES
--     Extra dependencies:          m pthread cudart_static -lpthread dl rt nppc nppial nppicc nppicom nppidei nppif nppig nppim nppist nppisu nppitc npps cublas cufft -L/usr/local/cuda/lib64 -L/usr/lib/x86_64-linux-gnu
--     3rdparty dependencies:
-- 
--   OpenCV modules:
--     To be built:                 aruco bgsegm bioinspired calib3d ccalib core cudaarithm cudabgsegm cudafeatures2d cudafilters cudaimgproc cudalegacy cudaobjdetect cudaoptflow cudastereo cudawarping cudev datasets dnn dnn_objdetect dpm face features2d flann freetype fuzzy gapi hdf hfs highgui img_hash imgcodecs imgproc java_bindings_generator line_descriptor ml objdetect optflow phase_unwrapping photo plot python2 python3 python_bindings_generator reg rgbd saliency shape stereo stitching structured_light superres surface_matching text tracking ts video videoio videostab xfeatures2d ximgproc xobjdetect xphoto
--     Disabled:                    cudacodec world
--     Disabled by dependency:      -
--     Unavailable:                 cnn_3dobj cvv java js matlab ovis sfm viz
--     Applications:                tests perf_tests examples apps
--     Documentation:               NO
--     Non-free algorithms:         YES
-- 
--   GUI: 
--     GTK+:                        YES (ver 3.22.30)
--       GThread :                  YES (ver 2.56.3)
--       GtkGlExt:                  NO
--     VTK support:                 NO
-- 
--   Media I/O: 
--     ZLib:                        /usr/lib/x86_64-linux-gnu/libz.so (ver 1.2.11)
--     JPEG:                        /usr/lib/x86_64-linux-gnu/libjpeg.so (ver 80)
--     WEBP:                        build (ver encoder: 0x020e)
--     PNG:                         /usr/lib/x86_64-linux-gnu/libpng.so (ver 1.6.34)
--     TIFF:                        /usr/lib/x86_64-linux-gnu/libtiff.so (ver 42 / 4.0.9)
--     JPEG 2000:                   /usr/lib/x86_64-linux-gnu/libjasper.so (ver 1.900.1)
--     OpenEXR:                     build (ver 1.7.1)
--     HDR:                         YES
--     SUNRASTER:                   YES
--     PXM:                         YES
--     PFM:                         YES
-- 
--   Video I/O:
--     DC1394:                      YES (ver 2.2.5)
--     FFMPEG:                      YES
--       avcodec:                   YES (ver 57.107.100)
--       avformat:                  YES (ver 57.83.100)
--       avutil:                    YES (ver 55.78.100)
--       swscale:                   YES (ver 4.8.100)
--       avresample:                YES (ver 3.7.0)
--     GStreamer:                   
--       base:                      YES (ver 1.14.1)
--       video:                     YES (ver 1.14.1)
--       app:                       YES (ver 1.14.1)
--       riff:                      YES (ver 1.14.1)
--       pbutils:                   YES (ver 1.14.1)
--     v4l/v4l2:                    linux/videodev2.h
-- 
--   Parallel framework:            pthreads
-- 
--   Trace:                         YES (with Intel ITT)
-- 
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
--   NVIDIA CUDA:                   YES (ver 10.0, CUFFT CUBLAS NVCUVID FAST_MATH)
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
--     numpy:                       /home/user/.local/lib/python2.7/site-packages/numpy/core/include (ver 1.16.1)
--     install path:                lib/python2.7/dist-packages/cv2/python-2.7
-- 
--   Python 3:
--     Interpreter:                 /usr/bin/python3 (ver 3.6.7)
--     Libraries:                   /usr/lib/x86_64-linux-gnu/libpython3.6m.so (ver 3.6.7)
--     numpy:                       /home/user/.local/lib/python3.6/site-packages/numpy/core/include (ver 1.16.1)
--     install path:                lib/python3.6/dist-packages/cv2/python-3.6
-- 
--   Python (for build):            /usr/bin/python2.7
--     Pylint:                      /home/user/anaconda3/envs/deep_learning_cv/bin/pylint (ver: 3.6.8, checks: 168)
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

```

### Installing mxnet deep learning library


### Installing other deep learning libraries
In this short tutorial, to prepare for all situations, we will be installing a wide range of deep learning and computer 
vision libraries, including:
* Tensorflow
* Keras
* PyTorch

```bash
(deep_learning_cv) $ conda install -c anaconda tensorflow-gpu mxnet keras pytorch

## Package Plan ##

  environment location: /home/user/anaconda3/envs/deep_learning_cv

  added / updated specs: 
    - keras
    - mxnet
    - opencv
    - pytorch
    - tensorflow-gpu


The following packages will be downloaded:
...
```
