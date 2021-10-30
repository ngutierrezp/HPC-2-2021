#ifndef IO_H
#define IO_H

int* read_raw(char* filename, int x, int y);
void write_hist(char* filename, int* hist, int* hist_shared,int Q);

#endif