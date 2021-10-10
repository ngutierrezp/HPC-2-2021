/*
    Integrantes:
        - Isidora Abasolo
        - Nicolás Gutiérrez
*/

// La explicación de las funciones se encuentran en los respectivos headers (.h)

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <ctype.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include "./incl/bomb.h"
#include "./incl/io.h"
#include "./incl/constants.h"
#include "./incl/colors.h"
#include "./incl/bomber.h"

int main(int argc, char *argv[])
{
    int opt, t = 0, N = 0, D = 0, *particles = (int *)malloc(sizeof(int));
    char *I = (char *)malloc(sizeof(char) * _MAX_STRING_SIZE), *O = (char *)malloc(sizeof(char) * _MAX_STRING_SIZE);


    // en caso de que no existan todos los argumentos
    if (argc != 11)
    {
        fprintf(stderr, RED "[ERROR]" reset "  El uso correcto es:\n\n\t " YEL "./bomb" reset " " GRN "-t" reset " numero_hebras " GRN "-N" reset " numero_celdas " GRN "-i" reset " input.txt " GRN "-o" reset " output.txt " GRN "-D" reset " debug " reset "\n\n");
        exit(EXIT_FAILURE);
    }
    // ##### segmento getopt ######
    /*
        -t: es el número de hebras.
        -N: es el n ́umero de celdas
        -i: es el archivo con el bombardeo
        -o: es el archivo de salida
        -D: especifica si imprime por stdout o no. 
            debug=1     imprime.
            debug=0     no imprime.

    */
    while ((opt = getopt(argc, argv, "t:N:i:o:D:")) != -1)
    {
        switch (opt)
        {
        case 't':
            t = atoi(optarg);
            break;
        case 'N':
            N = atoi(optarg);
            break;
        case 'i':
            strcpy(I, optarg);
            break;
        case 'o':
            strcpy(O, optarg);
            break;
        case 'D':
            D = atoi(optarg);
            break;
        default: /* '?' */
            fprintf(stderr, RED "[ERROR]" reset "  El uso correcto es:\n\n\t " YEL "./bomb" reset " " GRN "-t" reset " numero_hebras " GRN "-N" reset " numero_celdas " GRN "-i" reset " input.txt " GRN "-o" reset " output.txt " GRN "-D" reset " debug " reset "\n\n");
            exit(EXIT_FAILURE);
        }
    }
    float *new_vector = (float *)malloc(sizeof(float) * N);
    
    int **data = read_file(I, particles); // Lectura del archivo ->  matriz de enteros
    
    double begin_paral = omp_get_wtime();
    float *vector = bomber_openMP(data,*particles,N,t); // vector con energia
    double end_paral = omp_get_wtime();
    double time_parallel = (double)(end_paral - begin_paral); // tiempo de ejecución

    /*
        Para imprimir mediante la función niceprint, es necesario 
        copiar los datos de un nuevo vector de energias.
    */
    for (int i = 0; i < N; i++)
    {
        new_vector[i] = vector[i];
    }
    
    if (D == 1)
    {
        niceprint(N,new_vector); // la función solo se ejecutará si D == 1 (modo debug activado)
    }

    int index = get_index_max_energy(new_vector,N); // Se obtiene el indice de mayor valor de nergia
    write_file(O,N,new_vector,index); // Se imprime el archivo

    
    printf("\n\n");
    printf("Tiempo de de ejecución de la simulación \t:"GRN" %f seg."reset" \n",time_parallel);
    printf("Ejecutado correctamente. Se ha generado el archivo :"YEL" %s "reset".\n",O);


    /*
        ## Segmento de liberación de memoria ##
    */

    free(I);
    free(O);
    free(particles);
    free(vector);
    free(new_vector);
    for (int i = 0; i < *particles; i++)
    {
        free(data[i]);
    }
    free(data);

    return 0;
}
