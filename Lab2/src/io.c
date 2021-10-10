#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include "../incl/io.h"
#include "../incl/colors.h"

int **read_file(char *PATH_FILE, int *particles)
{

    int **datos;
    

    FILE *file = fopen(PATH_FILE, "r");
    if (!file)
    {
        fprintf(stderr, ERR" No se ha podido abrir el archivo. Verifique el nombre del archivo \n");
        exit(EXIT_FAILURE);
    }

    fscanf(file,"%d",particles);

    datos = (int**)malloc(sizeof(int*)*(*particles));
    

    for (int i = 0; i < *particles; i++)
    {
        datos[i] = (int*)malloc(sizeof(int)*2);
        
        fscanf(file,"%d %d",&datos[i][0],&datos[i][1]);
    }
    fclose(file);
    return datos;
}

void write_file(char *PATH_FILE, int N, float *vector, int index)
{

    FILE *file = fopen(PATH_FILE, "w");
    if (!file)
    {
        fprintf(stderr, ERR" No se ha podido crear el archivo. ¿El archivo ya existe? \n");
        exit(EXIT_FAILURE);
    }

    fprintf(file,"%d \t%f\n",index, vector[index]);

    for (int i = 0; i < N; i++)
    {
        fprintf(file,"%d \t%f\n",i, vector[i]);
    }

    fclose(file);
    
}
