#include "../incl/io.h"
#include <stdio.h>
#include <stdlib.h>

/*
    Función para leer una imagen de un archivo raw y 
    retorna un arreglo unidimensional de tipo unsigned int.
    @param filename Nombre del archivo.
    @param x Cantidad de columnas de la imagen.
    @param y Cantidad de filas de la imagen.
*/
int* read_raw(char* filename, int x, int y) 
{
    unsigned short int* image = (unsigned short int*) malloc(x * y * sizeof(unsigned short int));
    FILE* file = fopen(filename, "rb");
    fread(image, sizeof(unsigned int), x * y, file);
    fclose(file);

    int* image_int = (int*) malloc(x * y * sizeof(int));
    for (int i = 0; i < x * y; i++)
    {
        image_int[i] = (int)image[i];
    }
    free(image);
    return image_int;
}

