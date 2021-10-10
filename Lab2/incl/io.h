#ifndef IO_H
#define IO_H


/*
Descripción:    Función que lee un archivo .txt y genera una matriz de datos de partículas
                De la forma: 

                    Posición Impacto  | Energía de impacto
                            |                   |
                            v                   v

                            27                  602
                            23                  234
                            ...                 ...

                La posición de impacto corresponde al índice 0 y la energía al índice 1.

Parámetro PATH_FILE: Es un string que representa la dirección o nombre de un archivo .txt. 

Parámetro Particles: corresponde a un valor de memoria donde se almacenaran la cantidad de 
                    Partículas leídas en el archivo.

Salida:  Devuelve una matriz de datos de partículas donde la columna 0 es la posición de impacto
        y la columna 1 es la energía de impacto.
*/
int **read_file(char *PATH_FILE, int *particles);

/*
Descripción:    Función que genera un archivo que contiene la información listada de 
                las posiciones del material impactado y su energía.

Parámetro PATH_FILE: Es un string que representa la dirección o nombre de un archivo .txt. 

Parámetro vector:   Corresponde al vector de energías que representa al material donde
                    impactaron las partículas. 

Parámetro N: Corresponde a las divisiones del elemento en 1D. Es el largo de este elemento.

Parámetro Index: corresponde a la posición del elemento de mayor energía en el vector de energías.

Salida:  No posee salidas pero genera un archivo con el listado de posiciones y energía en esas posiciones.
*/
void write_file(char *PATH_FILE, int N, float *vector, int index);
#endif
