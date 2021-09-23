#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include "../incl/read.h"
#include "../incl/colors.h"

int **read_image(char *PATH_FILE, int M, int N)
{

    int size = N * M, file, value;
    int *buffer = (int *)malloc(sizeof(int) * size), **matriz = (int**)malloc(sizeof(int*)*N);

    file = open(PATH_FILE, O_RDONLY);
    if (file == -1)
    {
        fprintf(stderr, RED "[ERROR]" reset "  No se ha podido abrir el archivo. Verifique el nombre del archivo \n");
        exit(EXIT_FAILURE);
    }

    read(file, buffer, sizeof(int) * size);

    close(file);

    for (int i = 0; i < N; i++)
    {
        matriz[i] = (int*)malloc(sizeof(int) * M);
    }
    
    value = 0;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            matriz[i][j] = (int)buffer[value];
            value++;
        }
        
        
    }

    return matriz;
}