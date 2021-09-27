#ifndef READ_H
#define READ_H

int **read_image(char *PATH_FILE, int M, int N);
void write_image(int **matriz, char *OUTPUT_PATH, int T, int R);
void print_line_hough(int** H, int T, int R);

#endif