#include <cuda.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda_runtime_api.h>
#include "../incl/generate_hist.h"
#include <device_launch_parameters.h>

/*
    Kernel para generar histograma de una imagen en escala de grises.
    Tomando como largo del histograma el valor de cada pixel [0,255].

    @param I - Imagen en escala de grises en formato de vector de enteros.
    @param H - Histograma en formato de vector de enteros.
    @param n - Numero de filas de la imagen.
    @param m - Numero de columnas de la imagen.
    @param Q - Tamaño del histograma (256).

    @return - No posee retorno. El valor del histrograma es pasado por referencia.

*/
__global__ void histogram(int *I, int n, int m, int *H, int Q)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int value = I[id];

    if (id < n * m)
    {
        if (value < Q)
        {
            atomicAdd(&H[value], 1);
        }
    }
}


/*
    Kernel para generar histograma de una imagen en escala de grises.
    Tomando como largo del histograma el valor de cada pixel [0,255] utilizando 
    memoria compartida en CUDA.

    @param I - Imagen en escala de grises en formato de vector de enteros.
    @param H - Histograma en formato de vector de enteros.
    @param n - Numero de filas de la imagen.
    @param m - Numero de columnas de la imagen.
    @param Q - Tamaño del histograma (256).

    @return - No posee retorno. El valor del histrograma es pasado por referencia.

*/
__global__ void histogram_shared(int *I, int n, int m, int *H, int Q)
{
    #define BLOCK_SIZE 256
    __shared__ int shared[BLOCK_SIZE];

    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Se inicializ el vector de memoria compartida en 0.
    if (threadIdx.x == 0)
    {
        for (int i = 0; i < BLOCK_SIZE; i++)
        {
            shared[i] = 0;
        }
    }
    
    //Barrera
    __syncthreads();
    if (id < n*m)
    {
        int value = I[id];
        atomicAdd(&shared[value], 1);
    }
       //Barrera
    __syncthreads();

    if (threadIdx.x == 0)
    {
        for (int i = 0; i < BLOCK_SIZE; i++)
        {
            atomicAdd(&H[i], shared[i]);
        }
    }
}