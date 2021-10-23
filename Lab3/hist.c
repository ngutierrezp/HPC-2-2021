#include <stdio.h>
#include <stdlib.h>
#include "./incl/colors.h"
#include "./incl/constants.h"
#include "./incl/io.h"
#include <unistd.h>
#include <ctype.h>
#include <string.h>
#include <cuda.h>

// Función de suma en cuda con operación atomica.
__global__ void histogram(unsigned short int *a, unsigned short int *c, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned short int value = a[id];

    if (id < n) {
        atomicAdd(&c[value], 1 + c[value]);
    }
}

// Función de suma en cuda con operación atomica con memoria compartida.
__global__ void histogram_shared(unsigned short int *a, unsigned short int *c, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned short int value = a[id];

    __shared__ unsigned short int shared[256];

    if (id < n) {
        shared[value] = 1 + shared[value];
        atomicAdd(&c[value], shared[value]);
    }
}



// Suma los valores de un array en cuda con memoria compartida.
void sum_histogram(unsigned short int *a, unsigned short int *c, int n) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    histogram <<< grid_size, block_size >>> (a, c, n);
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
    int debug = 0, m = 0,n = 0, t = 0;
    char *i = (char*)malloc(sizeof(char)*_MAX_STRING_SIZE);
    char *o = (char*)malloc(sizeof(char)*_MAX_STRING_SIZE);
    int opt;
    if (argc != 13) {
        fprintf(stderr, RED "[ERROR]" reset "  El uso correcto es:\n\n\t " YEL "./hist" reset " " GRN "-i" reset " imagen_de_entrada.raw " GRN "-o" reset " imagen_de_salida.raw " GRN "-m" reset " tamañoX " GRN "-n" reset " tamañoY " GRN "-t" reset " numero_hebras " GRN "-d" reset " debug\n\n");
        exit(1);
    }
    
    while ((opt = getopt(argc, argv, "i:m:n:o:t:d:")) != -1) {
        switch (opt) {
            case 'i':
                strcpy(i,optarg);
                break;
            case 'm':
                m = atoi(optarg);
                break;
            case 'n':
                n = atoi(optarg);
                break;
            case 'o':
                strcpy(o,optarg);
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
    
    if (debug) {
        printf("\n\n");
        printf(GRN "Imagen de entrada: " reset "%s\n", i);
        printf(GRN "Tamaño de la imagen: " reset "%d x %d\n", m, n);
        printf(GRN "Archivo de salida: " reset "%s\n", o);
        printf(GRN "Numero de hebras: " reset "%d\n", t);
        printf(GRN "Debug: " reset "%d\n", debug);
        printf("\n\n");
    }

    // Se lee la imagen de entrada con la función read_raw()
    unsigned short int* image = read_raw(i, m, n);

    // Liberación de memoria
    free(i);
    free(o);
    free(image);
    
    return 0;
}
