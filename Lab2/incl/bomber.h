#ifndef BOMBER_H
#define BOMBER_H

/*
Descripción:    Función de simulación de bombardeo de partículas sobre un elemento en 1D.
                La función toma los datos leidos en forma de matriz:

                    Posición Impacto  | Energía de impacto
                            |                   |
                            v                   v

                            27                  602
                            23                  234
                            ...                 ...

                La posición de impacto corresponde al índice 0 y la energía al índice 1.
                La función crea un array correspondiente al elemento en 1D dividio en
                pequeñas posiciones donde impactan las partículas. La función retorna
                la cantidad de energía recibida por todas las partículas impactadas.

Parámetro Data: Corresponde a la matriz con la información leída del archivo de entrada.
                Contiene las posiciones y energía de impacto.

Parámetro Partículas: Corresponde al número de partículas que impactan.

Parámetro N: Corresponde a las divisiones del elemento en 1D. Es el largo de este elemento.

Parámetro Threads: Corresponde al número de hebras que se van a utilizar.

Salida:  Genera un array de float de tamaño N con la energía de las partículas distribuidas.

*/
float *bomber_openMP(int **data, int particles, int N, int threads);


/*
Descripción:    Función que obtiene el índice del máximo valor de energía del arreglo 
                de 1D correspondiente al material donde impactaron las partículas. 

Parámetro Vector:   Corresponde al vector de energías que representa al material donde
                    impactaron las partículas. 

Parámetro N: Corresponde a las divisiones del elemento en 1D. Es el largo de este elemento.

Salida:  Devuelve el indice del valor de mayor energía en el array de Energías. 

*/
int get_index_max_energy(float * vector, int N);

#endif
