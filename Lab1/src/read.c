#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include "../incl/read.h"
#include "../incl/colors.h"

int **read_image(char *PATH_FILE, int M, int N)
{

    int size = N * M, file, value;
    int *buffer = (int *)malloc(sizeof(int) * size), **matriz = (int **)malloc(sizeof(int *) * M);

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
        matriz[i] = (int *)malloc(sizeof(int) * N);
    }

    value = 0;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matriz[i][j] = (int)buffer[value];
            value++;
        }
    }

    return matriz;
}

void write_image(int **matriz, char *OUTPUT_PATH, int T, int R)
{

    int size = T * R, file, value=0;
    int *buffer = (int *)malloc(sizeof(int) * size);

    for (int i = 0; i < T; i++)
    {
        for (int j = 0; j < R; j++)
        {
            buffer[value] = (int) matriz[i][j];
            value++;
        }
            
    }
    

    file = open(OUTPUT_PATH, O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);

    if (file == -1)
    {
        fprintf(stderr, RED "[ERROR]" reset "  No se ha podido crear el archivo %s. Compruebe si ya existe. \n",OUTPUT_PATH);
        exit(EXIT_FAILURE);
    }
   
    write(file, buffer, size*sizeof(int));
   
    close(file);
   
    free(buffer);
   

}
