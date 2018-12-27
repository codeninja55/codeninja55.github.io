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

This is a post series as a reminder to myself as much as to help those who are considering building and setting up a 
similar system. Any comments or suggestions would be welcome as I am still a student learning about how to best 
configure these systems. 

This is a three part series that I will endeavour to update regularly as I discover better workflows.
1. Part 1 (this post): Hardware Drivers and System OS Installation.
2. Part 2: Development Environment and Library Installation
3. Part 3: Configuring Remote Access and Testing your new Development Environment 

## Hardware and System OS Installation
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
these configurations in the past. Also ensure you are installing Ubuntu 18.04 on the SSD that is empty, not the one that 
you used to install Windows 10. Usually this will be either `nvme0n1` or `nvme1n1`.

Additionally, I prefer to encrypt my Ubuntu OS whenever I use it so I also selected 'LVM' and 'Encryption'.

Once Ubuntu has finished installing, the first thing you will need to do is update the system to the latest versions of 
packages. To do this, run this in a terminal.

```bash
$ sudo apt update && sudo apt dist-upgrade 
```

Additionally, we will be using the latest kernel available to ensure the best support for the hardware of the PC. The 
Ubuntu 18.04 package install comes prepackaged with kernel 4.13, however, the latest at the time of writing was 4.20. 
To install this, we will using a third-party package called [ukuu](https://github.com/teejee2008/ukuu) to upgrade to the 
latest.

 

## Setting up the Ubuntu 18.04 for Hardware Compatibility
At this stage, we want to set up the Ubuntu to ensure all packages required for deep learning and GPU support will be 
ready. To do this, we will also be installing some additional useful packages.

```bash
$ sudo apt-add-repository -y ppa:teejee2008/ppa
$ sudo apt update
$ sudo apt install -y ukuu
$ ukuu --install --yes v4.20
```

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

## Jetbrains Toolbox and IDE
One of my favourite IDE to use during development it Jetbrain's various development editors - particularly PyCharm and 
IntelliJ IDEA. To easily install as many of them as possible, Jetbrains provides a 
[Toolbox application](https://www.jetbrains.com/toolbox/app/) you can use to easily install each individual IDE. To install the Toolbox app, create a script and use it to install.

```bash
$ touch jetbrains-toolbox.sh
$ nano jetbrains-toolbox.sh
```
* Create a shell script as believe using any editor.
* In this example, I have used the `nano` editor and then copied from below and used _Ctrl + Alt + V_ to paste.

#### `jetbrains-toolbox.sh`
```shell
function getLatestUrl() {
  USER_AGENT=('User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36')

  URL=$(curl 'https://data.services.jetbrains.com//products/releases?code=TBA&latest=true&type=release' -H 'Origin: https://www.jetbrains.com' -H 'Accept-Encoding: gzip, deflate, br' -H 'Accept-Language: en-US,en;q=0.8' -H "${USER_AGENT[@]}" -H 'Accept: application/json, text/javascript, */*; q=0.01' -H 'Referer: https://www.jetbrains.com/toolbox/download/' -H 'Connection: keep-alive' -H 'DNT: 1' --compressed | grep -Po '"linux":.*?[^\\]",' | awk -F ':' '{print $3,":"$4}'| sed 's/[", ]//g')
  echo $URL
}
getLatestUrl
FILE=$(basename ${URL})
DEST=$PWD/$FILE

actionMessage "Downloading Toolbox files"
wget -cO  ${DEST} ${URL} --read-timeout=5 --tries=0
echo ""
generalMessage "Download complete."

DIR="/opt/jetbrains-toolbox"
actionMessage "Installing to $DIR"

if mkdir ${DIR}; then
    tar -xzf ${DEST} -C ${DIR} --strip-components=1
fi

chmod -R +rwx ${DIR}
touch ${DIR}/jetbrains-toolbox.sh
echo "#!/bin/bash" >> $DIR/jetbrains-toolbox.sh
echo "$DIR/jetbrains-toolbox" >> $DIR/jetbrains-toolbox.sh
echo ""
ln -s ${DIR}/jetbrains-toolbox.sh /usr/local/bin/jetbrains-toolbox
chmod -R +rwx /usr/local/bin/jetbrains-toolbox
rm ${DEST}
```

```bash
$ sudo chmod +x jetbrains-toolbox.sh
$ sudo ./jetbrains-toolbox.sh
$ jetbrains-toolbox
```
* After you run `jetbrains-toolbox` for the first time, a desktop application shortcut will be added which you can 
use next time to run the tool.
* Make sure you install PyCharm Community or Professional after this by signing into your Jetbrains account and going
 to _Tools_ and pressing _Install_ for the PyCharm Professional or Community.

![JetBrains Toolbox App](/_images/2018-12-27-jetbrains_toolbox.PNG)

## Installing nVidia GPU drivers
Some of the deep learning libraries we will be installing later will use the GPU and CUDA to allow better processing of 
machine learning computations. To ensure they work properly, you must install the correct drivers for your GPU. In my 
case, the following will work.

### (1)
Check that your GPU is visible to the kernel via the PCI-e lanes:
```
$ lspci -nn | grep -i nvidia

01:00.0 VGA compatible controller [0300]: NVIDIA Corporation Device [10de:1e07] (rev a1)
01:00.1 Audio device [0403]: NVIDIA Corporation Device [10de:10f7] (rev a1)
01:00.2 USB controller [0c03]: NVIDIA Corporation Device [10de:1ad6] (rev a1)
01:00.3 Serial bus controller [0c80]: NVIDIA Corporation Device [10de:1ad7] (rev a1)

$ sudo lshw -numeric -C display

*-display                 
       description: VGA compatible controller
       product: NVIDIA Corporation [10DE:1E07]
       vendor: NVIDIA Corporation [10DE]
       physical id: 0
       bus info: pci@0000:01:00.0
       logical name: /dev/fb0
       version: a1
       width: 64 bits
       clock: 33MHz
       capabilities: pm msi pciexpress vga_controller bus_master cap_list rom fb
       configuration: depth=32 driver=nvidia latency=0 mode=3440x1440 visual=truecolor xres=3440 yres=1440
       resources: iomemory:600-5ff iomemory:600-5ff irq:178 memory:80000000-80ffffff memory:6000000000-600fffffff memory:6010000000-6011ffffff ioport:3000(size=128) memory:81000000-8107ffff
  *-display
       description: VGA compatible controller
       product: Intel Corporation [8086:3E98]
       vendor: Intel Corporation [8086]
       physical id: 2
       bus info: pci@0000:00:02.0
       version: 00
       width: 64 bits
       clock: 33MHz
       capabilities: pciexpress msi pm vga_controller bus_master cap_list rom
       configuration: driver=i915 latency=0
       resources: iomemory:600-5ff iomemory:400-3ff irq:145 memory:6013000000-6013ffffff memory:4000000000-400fffffff ioport:4000(size=64) memory:c0000-dffff

```

You will need to install some additional Linux packages for later so do them now.
```bash
$ sudo apt install -y gcc build-essential
```

### (2)
Next, Ubuntu comes with a default open-source driver for GPU called 'nouveau'. Allowing this driver to continue to 
work will stop the nVidia drivers from working. To blacklist this during boot up, create the following configuration 
file in `/etc/modprobe.d/blacklist-nouveau.conf`.
```bash
$ sudo touch /etc/modprobe.d/blacklist-nouveau.conf
$ sudo nano /etc/modprobe.d/blacklist-nouveau.conf
```

#### `blacklist-nouveau.conf`
```
blacklist nouveau
blacklist lbm-nouveau
options nouveau modeset=0
alias nouveau off
alias lbm-nouveau off
```

After creating the above conf file, ensure you update boot up process by running these commands and then rebooting.

```bash
$ sudo echo "options nouveau modeset=0" | sudo tee -a /etc/modprobe.d/nouveau-kms.conf
$ sudo update-initramfs -u
$ sudo update-grub
$ sudo reboot
```

[IMPORTANT !!!] After you have rebooted back to Ubuntu, ensure you follow these steps carefully.

### (3) 
Before you login to the `gdm3` X Window System, press _Ctl + Alt + F2_ to use the virtual console `ttyn2`. Login 
through the ttyn terminal:
  ```bash
  Ubuntu 18.04 CN55-GENIE-Ubuntu tty2
  CN55-GENIE-Ubuntu login: codeninja
  Password: ********
  $
  ```

### (4) 
Check that you have blacklisted 'nouveau' properly by running:
  ```bash
  $ lsmod | grep -i nouveau
  ```

You should see nothing be returned.

### (5)
We need to stop the display manager and drop to run level 3 next.
  ```bash
  $ sudo service gdm3 stop
  $ sudo service --status-all | grep -i gdm3
  
  [ - ]  gdm3
  
  $ sudo init 3
  ```
  \* Ensure that the `[ - ]` is showing for `gdm3` which means you have stopped the service properly.

### (6) 
We are going to use a PPA which will allow us to use the latest nVidia proprietary drivers.
  ```bash
  $ sudo add-apt-repository ppa:graphics-drivers/ppa
  $ sudo apt update
  ```

Before we install the driver, we will use Ubuntu to find out what is the recommended driver for our GPU.

  ```bash
  $ ubuntu-drivers devices
  
  == /sys/devices/pci0000:00/0000:00:01.0/0000:01:00.0 ==
  modalias : pci:v000010DEd00001E07sv00001043sd00008668bc03sc00i00
  vendor   : NVIDIA Corporation
  driver   : nvidia-driver-415 - third-party free recommended
  driver   : nvidia-driver-410 - third-party free
  driver   : xserver-xorg-video-nouveau - distro free builtin
  ```  
* The output above says that `nvidia-driver-415` is the recommended proprietary driver to use so we will install that. 
* You could also just install the driver using `sudo ubuntu-drivers autoinstall` which would install the recommended 
driver. 

Finally, we will install the nVidia driver and an additional utility called `nvidia-modprobe`. 
  ```bash
  $ sudo apt install -y nvidia-driver-415 nvidia-modprobe
  $ sudo reboot
  ```
* During installation, you will be asked to follow some prompts. Once the installation is done, ensure you reboot.

After installation, we can test if everything went well.
```bash
$ nvidia-modprobe
$ nvidia-smi

+-----------------------------------------------------------------------------+
| NVIDIA-SMI 415.25       Driver Version: 415.25       CUDA Version: 10.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce RTX 208...  Off  | 00000000:01:00.0  On |                  N/A |
| 19%   54C    P0    54W / 260W |   1476MiB / 10989MiB |      2%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      1570      G   /usr/lib/xorg/Xorg                           660MiB |
|    0      1793      G   /usr/bin/gnome-shell                         342MiB |
+-----------------------------------------------------------------------------+

```
* Please note I have `Processes` running because my GPU is being used to run the X Window System including the 
`gnome-shell` and `Xorg`.

## Installing CUDA Toolkit and cuDNN

When working with Deep Learning and GPU's, inevitably, you will need a utility framework provided by nVidia called 
[NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=runfilelocal) which provides a development environment for creating high performance GPU-accelerated 
applications. 

CUDA is a tool written mostly in C++ and to use it, you will generally be writing in the same language. However, 
Python libraries such as tensorflow and pytorch have provided a wrapping Application Programming Interface (API) 
where you can still use CUDA tensors and optimized operations to perform incredibly fast deep learning programming.

To install CUDA, you can either follow the thorough [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia
.com/cuda/cuda-installation-guide-linux/index.html) or just follow the steps below.

### (1) Downloading CUDA Installation Runfile
The first step we need to do is download the installation run file from this 
[site](https://developer.nvidia.com/cuda-downloads). From here, select the appropriate options as per below:

![NVIDIA CUDA Download page](/_images/2018-12-27-cuda_download.png)

Download the file by clicking the __Download (2.0 GB)__ button. Ensure you download it to `/home/<username>/Downloads`.

We will be using the run file to avoid installing the nVidia driver again as part of this process.

### (2) Installing CUDA from the runfile.

From here, we will need to logout of `gdm3` again and repeat step (3) from when we installed the nVidia driver.

Press _Ctl + Alt + F2_ to use the virtual console `ttyn2`. Login through the `ttyn` terminal:
  ```bash
  Ubuntu 18.04 CN55-GENIE-Ubuntu tty2
  CN55-GENIE-Ubuntu login: codeninja
  Password: ********
  $ sudo service gdm3 stop 
  $ sudo service --status-all | grep -i gdm3
    
  [ - ]  gdm3
    
  $ sudo init 3
  ```

Now that we have done this, we need to execute the runfile installation process. These commands will ensure you do 
not need to follow any prompts. 

```bash
$ cd $HOME/Downloads/
$ sudo ./cuda_10.0.130_410.48_linux.run --silent --toolkit --toolkitpath=/usr/local/cuda-10.0/ --samples
$ sudo ln -s /usr/local/cuda /usr/local/cuda-10.0
```
* The last line just adds a symbolic link to the CUDA installation path to allow you to install a newer version at a 
later date.

The following details the advanced options we used. 

| Action | Options Used | Explanation |
|--------|--------------|-------------|
| Silent installation | --silent | Required for any silent installation. Performs an installation with no further user-input and minimal command-line output based on the options provided below. Silent installations are useful for scripting the installation of CUDA. Using this option implies acceptance of the EULA. The following flags can be used to customize the actions taken during installation. At least one of --driver, --uninstall, --toolkit, and --samples must be passed if running with non-root permissions. |
 | | --driver | Install the CUDA Driver |
 | | --toolkit | Install the CUDA Toolkit. |
 | | --toolkitpath=\<path\> | Install the CUDA Toolkit to the <path> directory. If not provided, the default path of `/usr/local/cuda-10.0` is used. |
 | | --samples | Install the CUDA Samples |
 | | --samplespath=\<path\> | Install the CUDA Samples to the <path> directory. If not provided, the default path of `$HOME/NVIDIA_CUDA-10.0_Samples` is used. |
 
### (3) Post-installation steps
To ensure the system knows where your CUDA installation is, you will need to add some environment variables to your 
system. You can do this by adding these to your `$HOME/.bashrc` profile file.

```bash
$ echo '' >> $HOME/.bashrc 
$ echo '##### CUDA INSTALLATION VARIABLES #####' >> $HOME/.bashrc
$ echo 'export CUDA=/usr/local/cuda/bin' >> $HOME/.bashrc 
$ echo 'export CUDA_SAMPLES=$HOME/NVIDIA_CUDA-10.0_Samples/bin/x86_64/linux/release' >> $HOME/.bashrc 
$ echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> $HOME/.bashrc 
$ echo 'export PATH=$CUDA:$CUDA_SAMPLES:$PATH' >> $HOME/.bashrc 
$ source $HOME/.bashrc 
```
* I added the CUDA Samples variable also so you can easily run some of the samples later. After testing, you can 
remove this line.

### (4) Testing the CUDA installation

Now that we have installed and setup environment variables for CUDA, we will test the installation by running some of
 the samples that were installed with CUDA. Follow the following steps.
 
```bash
$ cd $HOME/NVIDIA_CUDA-10.0_Samples
$ make -j4
$ deviceQuery

deviceQuery Starting...

 CUDA Device Query \(Runtime API\) version (CUDART static linking)

Detected 1 CUDA Capable device\(s\)

Device 0: "GeForce RTX 2080 Ti"
  CUDA Driver Version / Runtime Version          10.0 / 10.0
  CUDA Capability Major/Minor version number:    7.5
  Total amount of global memory:                 10989 MBytes (11523260416 bytes)
  (68) Multiprocessors, ( 64) CUDA Cores/MP:     4352 CUDA Cores
  GPU Max Clock rate:                            1560 MHz (1.56 GHz)
  Memory Clock rate:                             7000 Mhz
  Memory Bus Width:                              352-bit
  L2 Cache Size:                                 5767168 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1024
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 3 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 10.0, CUDA Runtime Version = 10.0, NumDevs = 1
Result = PASS
```

If you have correctly installed CUDA and setup the environment variables, running the above will give you a similar 
output. 
