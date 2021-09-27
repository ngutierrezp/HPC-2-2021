#ifndef READ_H
#define READ_H


/*
Descripción: Dado una dirección de una imagen, lee el contenido y crea una matriz de
M x N.

parametro PATH_FILE: Es un string que representa la dirección o nombre de un archivo .raw

parametro M: Es el tamaño del eje X de la imagen .raw

parametro N: Es el tamaño del eje Y de la imagen .raw

Salida:  Devuelve una matriz de entreros que representa la imagen.
*/
int **read_image(char *PATH_FILE, int M, int N);


/*
Descripción: Función que convierte una matriz de enteros de hough en una imagen .raw.

parametro matriz: Es una matriz de enteros que representa la matriz de hough.

parametro T: Es un entero que representa la cantidad de angulos.

parametro R: Es un entero que representa la cantidad de distancias.

Salida:  No tiene un retorno. Genera un archivo .raw
*/
void write_image(int **matriz, char *OUTPUT_PATH, int T, int R);


/*
Descripción:    Funcion que imprime por pantalla la cantidad de lineas que encuentra.
                Esto se realiza recorriendo la matriz y detectando todo valor mayor a 0
                luego de una umbralización:

parametro H: Es una matriz de enteros que representa la matriz de hough.

parametro T: Es un entero que representa la cantidad de angulos.

parametro R: Es un entero que representa la cantidad de distancias.

Salida:  No tiene un retorno. Imprime por pantalla pares ordenados.
*/
void print_line_hough(int **H, int T, int R);

#endif