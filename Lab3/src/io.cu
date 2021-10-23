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
unsigned short int* read_raw(char* filename, int x, int y) 
{
    unsigned short int* image = (unsigned short int*) malloc(x * y * sizeof(unsigned short int));
    FILE* file = fopen(filename, "rb");
    fread(image, sizeof(unsigned int), x * y, file);
    fclose(file);
    return image;
}

