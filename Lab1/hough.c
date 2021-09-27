#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <ctype.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "./incl/read.h"
#include "./incl/constants.h"
#include "./incl/colors.h"
#include "./incl/transform.h"

int main(int argc, char *argv[])
{
    int M = 0, N = 0, T = 0, R = 0, U = 0, opt, offset, **H_matrix, **matrix, **SIMD_matriz;
    char *I = (char *)malloc(sizeof(char) * _MAX_STRING_SIZE), *O = (char *)malloc(sizeof(char) * _MAX_STRING_SIZE);

    // en caso de que no existan todos los argumentos
    if (argc != 15)
    {
        fprintf(stderr, RED "[ERROR]" reset "  El uso correcto es:\n\n\t " YEL "./hough" reset " " GRN "-i" reset " imagen.raw " GRN "-o" reset " hough.raw " GRN "-M" reset " tamañoX " GRN "-N" reset " tamañoY " GRN "-T" reset " numero_angulos " GRN "-R" reset " numero_distancias " GRN "-U" reset " umbral\n\n");
        exit(EXIT_FAILURE);
    }
    // ##### segmento getopt ######
    /*
        -I: imagen de entrada, con enteros (int) 0, o 255, en formato binario (raw).
        -O: imagen de salida con la transformada de Hough, con enteros (int), en formato binario
        -M: Núumero de filas de la imagen
        -N: Núumero de columnas de la imagen
        -T: Núumero de ángulos
        -R: Núumero de distancias
        -U: Umbral para detección de líneas

    */
    while ((opt = getopt(argc, argv, "i:o:M:N:T:R:U:")) != -1)
    {
        switch (opt)
        {
        case 'i':
            strcpy(I, optarg);
            break;
        case 'o':
            strcpy(O, optarg);
            break;
        case 'M':
            M = atoi(optarg);
            break;
        case 'N':
            N = atoi(optarg);
            break;
        case 'T':
            T = atoi(optarg);
            break;
        case 'R':
            R = atoi(optarg);
            break;
        case 'U':
            U = atoi(optarg);
            break;
        default: /* '?' */
            fprintf(stderr, RED "[ERROR]" reset "  El uso correcto es:\n\n\t " YEL "./hough" reset " " GRN "-i" reset " imagen.raw " GRN "-o" reset " hough.raw " GRN "-M" reset " tamañoX " GRN "-N" reset " tamañoY " GRN "-T" reset " numero_angulos " GRN "-R" reset " numero_distancias " GRN "-U" reset " umbral\n\n");
            exit(EXIT_FAILURE);
        }
    }

    //lectura de imagen
    matrix = read_image(I, M, N);

    //############ Segmento SECUENCIAL ############

    //ajuste de desplazamiento de la imagen
    offset = R / 2;

    // Transformada de Hough secuencial
    clock_t begin = clock();
    H_matrix = hough_transform(matrix, N, M, T, R, offset);
    clock_t end = clock();
    double time_secuential = (double)(end - begin) / CLOCKS_PER_SEC;

    // Umbralización
    umbralization(H_matrix, T, R, U);

    // Escritura la transformada secuencial
    write_image(H_matrix, O, T, R);

    //############ Segmento PARALELO ############

    //Transformada de Hough paralela
    clock_t SIMD_begin = clock();
    SIMD_matriz = SIMD_hough_transform(matrix, N, M, T, R, offset);
    clock_t SIMD_end = clock();
    double time_parallel = (double)(SIMD_end - SIMD_begin) / CLOCKS_PER_SEC;

    // Umbralización
    umbralization(SIMD_matriz, T, R, U);

    // Escritura la transformada paralela
    char out[_MAX_STRING_SIZE] = "SIMD_";
    strcat(out, O);
    write_image(SIMD_matriz, out, T, R);

    print_line_hough(H_matrix,T,R);

    
    printf("\n\n");
    printf("Tiempo de de ejecución de Hough secuencial \t:"GRN" %f seg."reset" \n",time_secuential);
    printf("Tiempo de de ejecución de Hough paralelo \t:"GRN" %f seg."reset" \n",time_parallel);
    printf("Ejecutado correctamente. Se han generado dos archivos :"YEL" %s "reset"y "YEL"%s.\n",O,out);

    free(I);
    free(O);
    free(matrix);
    free(H_matrix);
    free(SIMD_matriz);

    return 0;
}
