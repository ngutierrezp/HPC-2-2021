#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cpuid.h>
#include <mmintrin.h>
#include <xmmintrin.h>
#include "../incl/transform.h"

int **hough_transform(int **matriz, int N, int M, int T, int R, int offset)
{

    double delta_r = sqrt((M * M) + (N * N)) * 2 / (R);
    int **H = (int **)malloc(sizeof(int *) * T);

    // Creación de matriz de Hough
    for (int i = 0; i < T; i++)
    {
        H[i] = (int *)malloc(sizeof(int) * R);
    }

    for (int i = 0; i < T; i++)
    {
        for (int j = 0; j < R; j++)
        {
            H[i][j] = 0;
        }
    }

    // Transformada de Hogh
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (matriz[i][j] != 0)
            {

                for (int k = 0; k < T; k++)
                {
                    int r = (int)(i * cos(k * M_PI / T) + j * sin(k * M_PI / T)) / delta_r;

                    if (r < 0)
                    {
                        r = (int)abs(r + offset);
                    }
                    else
                    {
                        r = (int)r + offset;
                    }

                    H[k][r] = H[k][r] + 1;
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

int **SIMD_hough_transform(int **matriz, int N, int M, int T, int R, int offset)
{
    // PARALELIZACIÓN

    /* 
     La paralelización se realiza en 6 pasos

     ## Paso 1: Se obtienen los angulos a trabajar (k) y se multiplican por el Delta theta (Pi / T)
       
       k     * dT
       k+1   * dT
       k+2   * dT
       k+3   * dT
     
     ## Paso 2: Se calculan los cos y sen para los 4 angulos anteriores

     cos (k * dT)
     cos (k+1 * dT)
     cos (k+2 * dT)
     cos (k+3 * dT)

     -----

     sin (k * dT)
     sin (k+1 * dT)
     sin (k+2 * dT)
     sin (k+3 * dT)

     ## Paso 3: Se multiplican los valores de x(i) e y(j) a los valores de cos y sen

     i* cos (k * dT)
     i* cos (k+1 * dT)
     i* cos (k+2 * dT)
     i* cos (k+3 * dT)

     -----

     j* sin (k * dT)
     j* sin (k+1 * dT)
     j* sin (k+2 * dT)
     j* sin (k+3 * dT)

     ## Paso 4: Se suman los cos y sen

     i* cos (k * dT) + j* sin (k * dT)
     i* cos (k+1 * dT) + j* sin (k+1 * dT)
     i* cos (k+2 * dT) + j* sin (k+2 * dT)
     i* cos (k+3 * dT) + j* sin (k+3 * dT)

     ## Paso 5: Se multiplica el dR a los valores calculados
     
     ( i* cos (k * dT) + j* sin (k * dT) )       * dR
     ( i* cos (k+1 * dT) + j* sin (k+1 * dT) )   * dR
     ( i* cos (k+2 * dT) + j* sin (k+2 * dT) )   * dR
     ( i* cos (k+3 * dT) + j* sin (k+3 * dT) )   * dR

     ## EL ultimo paso es guardar los valores obtenidos en la matriz de la forma

     */

    float buffer[4] __attribute__((aligned(16))) = {0.0, 0.0, 0.0, 0.0};
    float delta_r = sqrt((M * M) + (N * N)) * 2 / (R);
    float delta_theta = M_PI / T;

    __m128 dR, dT, thetas, thetas_dT, SIMD_i, SIMD_j, SIMD_cos, SIMD_sin, SIMD_cos_i, SIMD_sin_j, SIMD_r, SIMD_r_dr;

    int **H = (int **)malloc(sizeof(int *) * T);

    for (int i = 0; i < T; i++)
    {
        H[i] = (int *)malloc(sizeof(int) * R);
    }

    for (int i = 0; i < T; i++)
    {
        for (int j = 0; j < R; j++)
        {
            H[i][j] = 0;
        }
    }

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (matriz[i][j] != 0)
            {

                if (T % 4 == 0)
                {
                    dR = _mm_set1_ps(delta_r);
                    dT = _mm_set1_ps(delta_theta);
                    SIMD_i = _mm_set1_ps((float)i);
                    SIMD_j = _mm_set1_ps((float)j);

                    // Transformada de Hough paralela
                    for (int k = 0; k < T / 4 * 4; k += 4)
                    {
                        float result[4] __attribute__((aligned(16)));
                        // ## PASO 1 ##
                        thetas = _mm_setr_ps((float)k, (float)k + 1, (float)k + 2, (float)k + 3);
                        thetas_dT = _mm_mul_ps(thetas, dT);
                        _mm_store_ps(result, thetas_dT);

                        // ## PASO 2 ##
                        SIMD_cos = _mm_setr_ps(cos(result[0]), cos(result[1]), cos(result[2]), cos(result[3]));
                        SIMD_sin = _mm_setr_ps(sin(result[0]), sin(result[1]), sin(result[2]), sin(result[3]));

                        // ## PASO 3 ##
                        SIMD_cos_i = _mm_mul_ps(SIMD_cos, SIMD_i);
                        SIMD_sin_j = _mm_mul_ps(SIMD_sin, SIMD_j);

                        // ## PASO 4 ##
                        SIMD_r = _mm_add_ps(SIMD_cos_i, SIMD_sin_j);

                        // ## PASO 5 ##
                        SIMD_r_dr = _mm_div_ps(SIMD_r, dR);

                        _mm_store_ps(buffer, SIMD_r_dr);

                        // Fin PARALELIZACIÓN

                        for (int b = 0; b < 4; b++)
                        {
                            if (buffer[b] < 0)
                            {
                                buffer[b] = abs(buffer[b] + offset);
                            }
                            else
                            {
                                buffer[b] = buffer[b] + offset;
                            }
                        }

                        // GUARDADO en MATRIZ H
                        H[k][(int)buffer[0]] = H[k][(int)buffer[0]] + 1;
                        H[k + 1][(int)buffer[1]] = H[k + 1][(int)buffer[1]] + 1;
                        H[k + 2][(int)buffer[2]] = H[k + 2][(int)buffer[2]] + 1;
                        H[k + 3][(int)buffer[3]] = H[k + 3][(int)buffer[3]] + 1;
                    }
                }
                else
                {
                    dR = _mm_set1_ps(delta_r);
                    dT = _mm_set1_ps(delta_theta);
                    SIMD_i = _mm_set1_ps((float)i);
                    SIMD_j = _mm_set1_ps((float)j);
                    for (int k = 0; k < T / 4 * 4; k += 4)
                    {
                        float result[4] __attribute__((aligned(16)));

                        // PARALELIZACIÓN
                       
                        // ## PASO 1 ##
                        thetas = _mm_setr_ps((float)k, (float)k + 1, (float)k + 2, (float)k + 3);
                        thetas_dT = _mm_mul_ps(thetas, dT);
                        _mm_store_ps(result, thetas_dT);

                        // ## PASO 2 ##
                        SIMD_cos = _mm_setr_ps(cos(result[0]), cos(result[1]), cos(result[2]), cos(result[3]));
                        SIMD_sin = _mm_setr_ps(sin(result[0]), sin(result[1]), sin(result[2]), sin(result[3]));

                        // ## PASO 3 ##
                        SIMD_cos_i = _mm_mul_ps(SIMD_cos, SIMD_i);
                        SIMD_sin_j = _mm_mul_ps(SIMD_sin, SIMD_j);

                        // ## PASO 4 ##
                        SIMD_r = _mm_add_ps(SIMD_cos_i, SIMD_sin_j);

                        // ## PASO 5 ##
                        SIMD_r_dr = _mm_div_ps(SIMD_r, dR);

                        _mm_store_ps(buffer, SIMD_r_dr);

                        // Fin PARALELIZACIÓN

                        for (int b = 0; b < 4; b++)
                        {
                            if (buffer[b] < 0)
                            {
                                buffer[b] = abs(buffer[b] + offset);
                            }
                            else
                            {
                                buffer[b] = buffer[b] + offset;
                            }
                        }

                        // GUARDADO en MATRIZ H
                        H[k][(int)buffer[0]] = H[k][(int)buffer[0]] + 1;
                        H[k + 1][(int)buffer[1]] = H[k + 1][(int)buffer[1]] + 1;
                        H[k + 2][(int)buffer[2]] = H[k + 2][(int)buffer[2]] + 1;
                        H[k + 3][(int)buffer[3]] = H[k + 3][(int)buffer[3]] + 1;
                    }
                    for (int k = T / 4 * 4; k < T; k++)
                    {
                        // Dado que como T no es multiplo de 4, a lo más 3
                        // iteraciones se realizarán de forma secuencial.

                        int r = (int)(i * cos(k * M_PI / T) +
                                      j * sin(k * M_PI / T)) /
                                delta_r;
                        if (r < 0)
                        {
                            r = (int)abs(r + offset);
                        }
                        else
                        {
                            r = (int)r + offset;
                        }

                        H[k][r] = H[k][r] + 1; 
                    }
                }
            }
        }
    }
    return H;
}