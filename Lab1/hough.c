#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <ctype.h>
#include <string.h>
#include "./incl/read.h"
#include "./incl/constants.h"
#include "./incl/colors.h"
#include "./incl/transform.h"

int main(int argc, char *argv[])
{
    int M = 0, N = 0, T = 0, R = 0, U = 0, opt, **H_matrix, **matrix;
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

    matrix = read_image(I, M, N);
    printf("Caso1: \n");
    H_matrix = SIMD_hough_transform(matrix,N,M,T,R);
    printf("Caso2: \n");
    umbralization(H_matrix,T,R,U);
    printf("Caso3: \n");
    write_image(H_matrix,O,T,R);


    free(I);
    free(O);
    free(matrix);
    free(H_matrix);


    printf("Ejecutado correctamente...\n");

    return 0;
}
