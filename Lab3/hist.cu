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
#include "./incl/generate_hist.h"
#include <device_launch_parameters.h>

/*
    ############ IMPORTANTE ############
    Leer el apartado de importancia en el readme.md
*/



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
    cudaEvent_t start, stop, star_shared, stop_shared;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&star_shared);
    cudaEventCreate(&stop_shared);

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
    
    
    // Obtención de las versiones de cuda.
    int *driver = (int *)malloc(sizeof(int));
    int *runtime = (int *)malloc(sizeof(int));
    cudaRuntimeGetVersion(runtime);
    cudaDriverGetVersion(driver);
    

    
    printf("\n\n");
    printf(GRN "Imagen de entrada: " reset "%s\n", i);
    printf(GRN "Tamaño de la imagen: " reset "%d x %d\n", m, n);
    printf(GRN "Archivo de salida: " reset "%s\n", o);
    printf(GRN "Numero de hebras: " reset "%d\n", t);
    printf(GRN "Debug: " reset "%d\n", debug);
    printf(GRN "Driver Version: " reset "%d\n", *driver);
    printf(GRN "Runtime Version: " reset "%d\n", *runtime);
    printf("\n\n");

    if (t > 1024)
    {
        printf(YEL "[WARNING]" reset " El numero de hebras por bloque no puede ser mayor a 1024.\n");
        printf("Se ha limitado el numero de hebras a 1024...\n\n\n");
        t = 1024;
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
        histo_shared[i] = 0;
    }

    cudaMalloc((void **)&histo_cuda, Q * sizeof(int));
    cudaMalloc((void **)&histo_cuda_shared, Q * sizeof(int));
    cudaMalloc((void **)&image_cuda, m * n * sizeof(int));
    cudaMemcpy(image_cuda, image, m * n * sizeof(int), cudaMemcpyHostToDevice);


    // ####################################################################
    // #                  Histograma en cuda                              #
    // ####################################################################


    // Empieza el evento
    cudaEventRecord(start);

    // Se copia el histograma a device
    cudaMemcpy(histo_cuda, histo, Q * sizeof(int), cudaMemcpyHostToDevice);


    // Se llama el kernel de histograma con operación atomica.
    histogram<<<(int)ceil((m * n) / t), t>>>(image_cuda, m, n, histo_cuda, Q);
    cudaDeviceSynchronize();

    // Se copia el histograma de device a host
    cudaMemcpy(histo, histo_cuda, Q * sizeof(int), cudaMemcpyDeviceToHost);
    
    //Termina el evento
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float timeinmilliseconds = 0;
    cudaEventElapsedTime(&timeinmilliseconds, start, stop); // Tiempo en milisegundos


    // ####################################################################
    // #             Histograma en cuda con memoria compartida            #
    // ####################################################################


    // Empieza el evento
    cudaEventRecord(star_shared);

    // Se copia el histograma a device
    cudaMemcpy(histo_cuda_shared, histo_shared, Q * sizeof(int), cudaMemcpyHostToDevice);

    // Kernel de histograma con memoria compartida
    histogram_shared<<<(int)ceil((m * n) / t), t>>>(image_cuda, m, n, histo_cuda_shared, Q);
    cudaDeviceSynchronize();

    // Se copia la memoria de histograma a la memoria en host.
    cudaMemcpy(histo_shared, histo_cuda_shared, Q * sizeof(int), cudaMemcpyDeviceToHost);

    //Termina el evento
    cudaEventRecord(stop_shared);
    cudaEventSynchronize(stop_shared);
    float timeinmilliseconds_shared = 0;
    cudaEventElapsedTime(&timeinmilliseconds_shared, star_shared, stop_shared); // Tiempo en milisegundos

    // ####################################################################



    // Print de histogramas en host.
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
    write_hist(o, histo, histo_shared, Q);
    printf("Histograma generado en: " GRN "%s\n" reset, o);
    printf("Tiempo en generar histograma: " GRN "%f\n" reset, timeinmilliseconds);
    printf("Tiempo en generar histograma con memoria compartida: " GRN "%f\n", timeinmilliseconds_shared);
    printf("\n\n");
    

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
