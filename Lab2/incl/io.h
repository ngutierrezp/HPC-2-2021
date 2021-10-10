#ifndef IO_H
#define IO_H


/*
Descripción:    Función que lee un archivo .txt y genera una matriz de datos de particulas
                De la forma: 

                    posición impacto  | Energia de impacto
                            |                   |
                            v                   v

                            27                  602
                            23                  234
                            ...                 ...

                La posición de impacto corresponde al indice 0 y la energia al indice 1.

parametro PATH_FILE: Es un string que representa la dirección o nombre de un archivo .txt

parametro particles: corresponde a un valor de memoria donde se almacenaran la cantidad de 
                    Particulas leidas en el archivo.

Salida:  Devuelve una matriz de datos de particulas donde la columna 0 es la posición de impacto
        y la columna 1 es la energia de impacto.
*/
int **read_file(char *PATH_FILE, int *particles);

/*
Descripción:    Función que genera un archivo que contiene la información listada de 
                las posiciones del material impactado y su energia.

parametro PATH_FILE: Es un string que representa la dirección o nombre de un archivo .txt

parametro vector:   Corresponde al vector de energias que representa al material donde
                    impactaron las particulas.

parametro N: Corresponde a las divisiones del elemento en 1D. Es el largo de este elemento.

parametro index: corresponde a la posición del elemento de mayor energia en el vector de energias.

Salida:  No posee salidas pero genera un archivo con el listado de posiciones y enegia en esas posiciones.
*/
void write_file(char *PATH_FILE, int N, float *vector, int index);
#endif