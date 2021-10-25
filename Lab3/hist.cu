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

/*
    ############ IMPORTANTE ############
    Leer el apartado de importancia en el readme.md
*/

//Función de suma en cuda con operación atomica.
__global__ void histogram(int *I, int n, int m, int *H, int Q)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int value = I[id];

    // Se obtiene el valor de la imagen en la posición id
    printf("mi id:\t%d - value :\t%d  \n", id, (int)value);

    if (id < n * m)
    {
        if (value < Q)
        {
            atomicAdd(&H[value], 1);
        }
    }
}

//Función que crea el histograma en cuda con memoria compartida.
__global__ void histogram_shared(int *I, int n, int m, int *H, int Q)
{
    #define BLOCK_SIZE 256
    __shared__ int shared[BLOCK_SIZE];

    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Se inicializ el vector de memoria compartida en 0.
    if (threadIdx.x < n*m)
    {
        shared[threadIdx.x] = 0;
    }
    //Barrera
    __syncthreads();
    if (id < Q)
    {
        int value = I[id];
        atomicAdd(&shared[value], 1);
    }
       //Barrera
    __syncthreads();

    if (threadIdx.x < n*m)
    {
        atomicAdd(&H[threadIdx.x], shared[threadIdx.x]);
    }

    
    
    
}

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
    cudaError_t status;
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
    int *driver = (int *)malloc(sizeof(int));
    int *runtime = (int *)malloc(sizeof(int));

    status = cudaRuntimeGetVersion(runtime);
    printf("%s \n", cudaGetErrorString(status));
    status = cudaDriverGetVersion(driver);
    printf("%s \n", cudaGetErrorString(status));

    if (debug)
    {
        printf("\n\n");
        printf(GRN "Imagen de entrada: " reset "%s\n", i);
        printf(GRN "Tamaño de la imagen: " reset "%d x %d\n", m, n);
        printf(GRN "Archivo de salida: " reset "%s\n", o);
        printf(GRN "Numero de hebras: " reset "%d\n", t);
        printf(GRN "Debug: " reset "%d\n", debug);
        printf(GRN "Driver Version: " reset "%d\n", *driver);
        printf(GRN "Runtime Version: " reset "%d\n", *runtime);
        printf("\n\n");
    }

    int *histo_cuda;
    int *histo_cuda_shared;
    int *image_cuda;

    int *image = read_raw(i, m, n); // lectura de la imagen de entrada.
    int *histo = (int *)malloc(sizeof(int) * Q);
    int *histo_shared = (int *)malloc(sizeof(int) * Q);
    // Se inicializa el array con 0.
    for (int i = 0; i < Q; i++)
    {
        histo[i] = 0;
    }

    cudaMalloc((void **)&histo_cuda, Q * sizeof(int));
    cudaMalloc((void **)&histo_cuda_shared, Q * sizeof(int));
    cudaMalloc((void **)&image_cuda, m * n * sizeof(int));

    // Se copia la imagen de entrada a la memoria en cuda.
    cudaMemcpy(image_cuda, image, m * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(histo_cuda, histo, Q * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(histo_cuda_shared, histo_shared, Q * sizeof(int), cudaMemcpyHostToDevice);

    // Se llama el kernel de histograma con operación atomica.
    histogram<<<(int)ceil((m * n) / t), t>>>(image_cuda, m, n, histo_cuda, Q);
    histogram_shared<<<(int)ceil((m * n) / t), t>>>(image_cuda, m, n, histo_cuda_shared, Q);
    cudaDeviceSynchronize();
    // Se copia la memoria de histograma a la memoria en host.
    cudaMemcpy(histo, histo_cuda, Q * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(histo_shared, histo_cuda_shared, Q * sizeof(int), cudaMemcpyDeviceToHost);

    // Print de histograma en host.
    if (debug)
    {
        printf(GRN "Histograma en host:\n" reset);
        printf("Valor\tHisto\tHisto_shared\n");
        for (int i = 0; i < Q; i++)
        {
            printf("%d\t->\t%d\t%d\n", i, histo[i],histo_shared[i]);
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
    free(driver);
    free(runtime);
    cudaFree(histo_cuda);
    cudaFree(image_cuda);

    return 0;
}
