#ifndef GENERATE_HIST_H
#define GENERATE_HIST_H

__global__ void histogram(int *I, int n, int m, int *H, int Q);
__global__ void histogram_shared(int *I, int n, int m, int *H, int Q);



#endif