# Lab 3 : HPC

Realizado por : 
- Isidora Abasolo
- Nicolás Gutiérrez


En este Tercer laboratorio se verá SIMT - CUDA

## Requisitos

- GPU Nvidia compatible con CUDA.
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
export PATH=$PATH:/usr/local/cuda-11/bin
# or 
export PATH=$PATH:/usr/local/cuda/bin
```

Luego reiniciar la bash y para probar que funciona bien : 

```sh
nvcc --version
```
