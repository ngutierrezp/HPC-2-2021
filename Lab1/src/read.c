#include <stdio.h>
#include <stdlib.h>
#include <curses.h>
#include "../incl/read.h"

void read_image(char *PATH)
{
    FILE *image_raw;
    
    int **matriz_image, test;

    int i, j, rows, colums;

    //i read dimension image

     sscanf(PATH,"%*[^-]-%dx%d",
                &rows,
                &colums);

    printf("size image : [X:%d , Y:%d]\n",rows,colums);

    //i create dinamic rows
    matriz_image = (int **)malloc(rows * sizeof(int *));

    //i create dinamic colums
    for (i = 0; i < rows; i++)
    {

        matriz_image[i] = (int *)malloc(colums * sizeof(int));
    }

    //i open image raw
    image_raw = fopen(PATH, "rb");

    //i copy values to matriz_image
    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < colums; j++)
        {

            
            fscanf(image_raw, "%d", &test);
            *(*(matriz_image + i) + j) = test;
            
        }
    }
    printf("imprimiendo matriz.... \n");
    //i print matriz
    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < colums; j++)
        {
            
            printf("%d ", *(*(matriz_image + i) + j));
            
        }
        printf("\n");
    }


}