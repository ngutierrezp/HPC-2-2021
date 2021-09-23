#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../incl/transform.h"


void hough_transform(int** matriz, int N, int M, int T){

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            if (matriz[i][j] != 0)
            {
                for (int k = 0; k < T; k++)
                {
                    float r = i * cos(k * M_PI/T) +  j * sen(k * M_PI/T);
                    r = r / 
                }
                
                
                    
            }
            
        }
        
    }
    

    

}