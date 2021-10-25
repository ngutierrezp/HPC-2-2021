# Lab 3 : HPC

Realizado por : 
- Isidora Abasolo
- Nicolás Gutiérrez


En este Tercer laboratorio se verá SIMT - CUDA

## Importante

    
El programa fue desarrollado con el siguiente sistema:
- Sistema Operativo: Ubuntu 20.04
- Versión de Cuda: 11.5
- Versión de Cuda Toolkit: 11.5
- Nvidia driver version : 470.48
- gcc version 9.3.0 (Ubuntu 9.3.0-17ubuntu1~20.04)

## Datos de Arquitectura

Resultado de diveseQuery:
    
```sh
./deviceQuery Starting...
CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "NVIDIA GeForce GTX 970"
CUDA Driver Version / Runtime Version          11.5 / 11.5
CUDA Capability Major/Minor version number:    5.2
Total amount of global memory:                 4040 MBytes (4236181504 bytes)
(13) Multiprocessors, (128) CUDA Cores/MP:     1664 CUDA Cores
GPU Max Clock rate:                            1253 MHz (1.25 GHz)
Memory Clock rate:                             3505 Mhz
Memory Bus Width:                              256-bit
L2 Cache Size:                                 1835008 bytes
Maximum Texture Dimension Size (x,y,z)         1D=(65536), 2D=(65536, 65536), 3D=(4096, 4096, 4096)
Maximum Layered 1D Texture Size, (num) layers  1D=(16384), 2048 layers
Maximum Layered 2D Texture Size, (num) layers  2D=(16384, 16384), 2048 layers
Total amount of constant memory:               65536 bytes
Total amount of shared memory per block:       49152 bytes
Total number of registers available per block: 65536
Warp size:                                     32
Maximum number of threads per multiprocessor:  2048
Maximum number of threads per block:           1024
Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
Maximum memory pitch:                          2147483647 bytes
Texture alignment:                             512 bytes
Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
Run time limit on kernels:                     Yes
Integrated GPU sharing Host Memory:            No
Support host page-locked memory mapping:       Yes
Alignment requirement for Surfaces:            Yes
Device has ECC support:                        Disabled
Device supports Unified Addressing (UVA):      Yes
Device supports Compute Preemption:            No
Supports Cooperative Kernel Launch:            No
Supports MultiDevice Co-op Kernel Launch:      No
Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
Compute Mode:
    < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 11.5, CUDA Runtime Version = 11.5, NumDevs = 1, Device0 = NVIDIA GeForce GTX 970
Result = PASS
```

## Requisitos

- GPU Nvidia compatible con CUDA.
- Driver Nvidia instalado > 450.80.02 ( ver [tabla de compatibilidad](https://docs.nvidia.com/deploy/cuda-compatibility/))
- compilador `gcc`
- compilador `nvcc`

## Instalación de CUDA y compilador nvcc

Primeramente se debe verificar que el pc es compatible con CUDA:
```sh
lspci | grep -i nvidia
```
Si se obtiene un resultado del comando anterior, es posible instalar CUDA.

Luego para instalar CUDA (para OS en base a debian / Ubuntu 20+ / Mint 20+):
```sh
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin

sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600

wget https://developer.download.nvidia.com/compute/cuda/11.5.0/local_installers/cuda-repo-ubuntu2004-11-5-local_11.5.0-495.29.05-1_amd64.deb

sudo dpkg -i cuda-repo-ubuntu2004-11-5-local_11.5.0-495.29.05-1_amd64.deb

sudo apt-key add /var/cuda-repo-ubuntu2004-11-5-local/7fa2af80.pub

sudo apt-get update

sudo apt-get -y install cuda

```

Ahora es necesario agregar el compilador al `PATH`.



```sh
# para bash original de Ubutu:
nano ~/.bashrc

# Para zsh
nano ~/.zshrc
```

Agregar la siguiente linea al final del archivo:
```sh
export PATH=$PATH:/usr/local/cuda-11.5/bin
# or 
export PATH=$PATH:/usr/local/cuda/bin
```

Luego reiniciar la bash y para probar que funciona bien : 

```sh
nvcc --version
```
