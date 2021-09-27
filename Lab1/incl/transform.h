#ifndef TRANSFORM_H
#define TRANSFORM_H

int **hough_transform(int **matriz, int N, int M, int T, int R,int offset);
void umbralization(int **matriz, int T, int R, int U);
int **SIMD_hough_transform(int **matriz, int N, int M, int T, int R, int offset);


#endif