#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../incl/transform.h"

int ** hough_transform(int **matriz, int N, int M, int T, int R)
{

    double delta_r = sqrt( (M * M) + (N * N)) / (2*R);
    int ** H =(int**)malloc(sizeof(int*)*T);

    // Creación de matriz de Hough
    for (int i = 0; i < T; i++)
    {
        H[i] = (int*)malloc(sizeof(int)*R);
    }

    for (int i = 0; i < T; i++)
    {
        for (int j = 0; j < R; j++)
        {
            H[i][j] = 0;
        }
        
    }
    
    // Transformada de Hogh
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            if (matriz[i][j] != 0)
            {
                for (int k = 0; k < T; k++)
                {
                    int r = (int)i * cos(k * M_PI / T) + j * sin(k * M_PI / T);
                    r = (int) r * delta_r;
                    
                    H[k][r] = H[k][r] + 1; // H(theta_i,r_j)
                }
            }
        }
    }
    return H;
}

void umbralization(int **matriz, int T, int R, int U)
{

    for (int i = 0; i < T; i++)
    {
        for (int j = 0; j < R; j++)
        {
            if (matriz[i][j] > U)
            {
                matriz[i][j] = 255;
            }
            else
            {
                matriz[i][j] = 0;
            }
        }
    }
}