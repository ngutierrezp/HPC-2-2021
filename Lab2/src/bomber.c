#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "../incl/bomber.h"

float *bomber_openMP(int **data, int particles, int N, int threads)
{

    float result, MIN_ENERGY = pow(10, -3) / N;
    float *vector = (float *)malloc(sizeof(float) * N);

    for (int i = 0; i < N; i++)
    {
        vector[i] = 0;
    }

    omp_set_num_threads(threads);

    if (threads > particles)
    {
        #pragma omp parallel private(result)
        {
        #pragma omp for schedule(guided,2)

            for (int j = 0; j < particles; j++)
            {
                for (int i = 0; i < N; i++)
                {
                    result =((pow(10, 3) * data[j][1]) / (N * sqrt(abs(data[j][0] - i) + 1)));
                    if (result > MIN_ENERGY)
                    {
                        #pragma omp critical
                        {
                            vector[i] =  vector[i] + result;
                        }
                    }
                }
            }
        }
    }
    else
    {
        int j = 0, my_j=0;
        #pragma omp parallel private(my_j,result)
        {
            while (j < particles)
            {
                #pragma omp critical
                {
                    my_j = j;
                    j++;
                }
                result = 0;
                for (int i = 0; i < N; i++)
                {
                    result = ((pow(10, 3) * data[my_j][1]) / (N * sqrt(abs(data[my_j][0] - i) + 1)));
                    if (result > MIN_ENERGY)
                    {
                        #pragma omp critical
                        {
                            vector[i] = vector[i] + result;
                        }
                    }
                }
            }
        }
    }
    return vector;
}
int get_index_max_energy(float * vector, int N){

    int max_value = -1,max_index=-1;


    for (int i = 0; i < N; i++)
    {
        if (max_value < vector[i])
        {
            max_value = vector[i];
            max_index = i;
        }
        
    }
    return max_index;
    
}