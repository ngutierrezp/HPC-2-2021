#include <cuda.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "./incl/io.h"
#include "./incl/colors.h"
#include "./incl/constants.h"
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

//Función de suma en cuda con operación atomica.
__global__ void histogram(int *I, int n, int m, int *H, int Q)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    printf(" mi id: %d\n", id);

    int value = H[id];

    if (id < n * m)
    {
        if (value < Q)
        {
            atomicAdd(&H[value], 1);
        }
    }
}

// Función de suma en cuda con operación atomica con memoria compartida.
// __global__ void histogram_shared( int *I, int n, int m,  int *H, int Q)
// {
//     int id = blockIdx.x * blockDim.x + threadIdx.x;

//      int value = I[id];

//     // define shared memory
//     __shared__  int temp[shared[Q]];

//     if (id < n)
//     {
//         if(value < Q)
//         {
//             atomicAdd(&shared[value], 1);
//         }

//     }
// }

int main(int argc, char *argv[])
{
    /*
    Se genera el segmento getotp con los parametros de entrada.
      -i: es la imagen de entrada
      -m: es el número de filas de la imagen
      -n: es el número de columnas de la imagen
      -o: es el archivo texto con los histogramas
      -t: especifica el número de hebras por bloque
      -d: debug=0, no imprime por stdout, debug=1.
    */
    // segmento getotp
    int debug = 0, m = 0, n = 0, t = 0;
    char *i = (char *)malloc(sizeof(char) * _MAX_STRING_SIZE);
    char *o = (char *)malloc(sizeof(char) * _MAX_STRING_SIZE);
    int opt;
    int Q = 256; // Se define Q por defecto para escala de grisis (0,255)
    if (argc != 13)
    {
        fprintf(stderr, RED "[ERROR]" reset "  El uso correcto es:\n\n\t " YEL "./hist" reset " " GRN "-i" reset " imagen_de_entrada.raw " GRN "-o" reset " imagen_de_salida.raw " GRN "-m" reset " tamañoX " GRN "-n" reset " tamañoY " GRN "-t" reset " numero_hebras " GRN "-d" reset " debug\n\n");
        exit(1);
    }

    while ((opt = getopt(argc, argv, "i:m:n:o:t:d:")) != -1)
    {
        switch (opt)
        {
        case 'i':
            strcpy(i, optarg);
            break;
        case 'm':
            m = atoi(optarg);
            break;
        case 'n':
            n = atoi(optarg);
            break;
        case 'o':
            strcpy(o, optarg);
            break;
        case 't':
            t = atoi(optarg);
            break;
        case 'd':
            debug = atoi(optarg);
            break;
        default:
            fprintf(stderr, RED "[ERROR]" reset "  El uso correcto es:\n\n\t " YEL "./hist" reset " " GRN "-i" reset " imagen_de_entrada.raw " GRN "-o" reset " imagen_de_salida.raw " GRN "-m" reset " tamañoX " GRN "-n" reset " tamañoY " GRN "-t" reset " numero_hebras " GRN "-d" reset " debug\n\n");
            exit(EXIT_FAILURE);
        }
    }

    if (debug)
    {
        printf("\n\n");
        printf(GRN "Imagen de entrada: " reset "%s\n", i);
        printf(GRN "Tamaño de la imagen: " reset "%d x %d\n", m, n);
        printf(GRN "Archivo de salida: " reset "%s\n", o);
        printf(GRN "Numero de hebras: " reset "%d\n", t);
        printf(GRN "Debug: " reset "%d\n", debug);
        printf("\n\n");
    }

    
    int *histo_cuda;
    int *image_cuda;

    int *image = read_raw(i, m, n); // lectura de la imagen de entrada.
    int *histo = (int *)malloc(sizeof(int) * Q);

    cudaMalloc((void **)&histo_cuda, Q * sizeof(int));
    cudaMalloc((void **)&image_cuda, m * n * sizeof(int));

    // Se copia la imagen de entrada a la memoria en cuda.
    cudaMemcpy(image_cuda, image, m * n * sizeof(int), cudaMemcpyHostToDevice);
    // Se copia la memoria de hisograma a la memoria en cuda.
    cudaMemcpy(histo_cuda, histo, Q * sizeof(int), cudaMemcpyHostToDevice);

    // Se llama el kernel de histograma con operación atomica.
    histogram<<<(m * n) / t, t>>>(image_cuda, m, n, histo_cuda, Q);

    // Se copia la memoria de histograma a la memoria en host.
    cudaMemcpy(histo, histo_cuda, Q * sizeof(int), cudaMemcpyDeviceToHost);

    // Print de histograma en host.
    if (debug)
    {
        printf(GRN "Histograma en host:\n" reset);
        for (int i = 0; i < Q; i++)
        {
            printf("%d: %d\n", i, histo[i]);
        }
        printf("\n\n");
    }

    // Se escribe el histograma en el archivo de salida.
    // write_raw(o, histo, Q);

    // Liberación de memoria
    free(i);
    free(o);
    free(image);
    free(histo);
    cudaFree(histo_cuda);
    cudaFree(image_cuda);

    return 0;
}
