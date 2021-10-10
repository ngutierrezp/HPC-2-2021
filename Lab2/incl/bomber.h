#ifndef BOMBER_H
#define BOMBER_H

/*
Descripción:    Función de simulación de bombardeo de particulas sobre un elemento en 1D.
                La función toma los datos leidos en forma de matriz:

                    posición impacto  | Energia de impacto
                            |                   |
                            v                   v

                            27                  602
                            23                  234
                            ...                 ...

                La posición de impacto corresponde al indice 0 y la energia al indice 1.
                La función crea un array correspondiente al elemento en 1D dividio en
                pequeñas posiciones donde impactan las particulas. La función retorna
                la cantidad de energia recibida por todas las particulas impactadas.

parametro data: Corresponde a la matriz con la información leida del archivo de entrada.
                Contiene las posiciones y energia de impacto.

parametro particles: Corresponde al numero de particulas que impactan.

Parametro N: Corresponde a las divisiones del elemento en 1D. Es el largo de este elemento.

parametro threads: Corresponde al numero de hebras que se van a utilizar.

Salida:  Genera un array de float de tamaño N con la energia de las particulas distribuidas.

*/
float *bomber_openMP(int **data, int particles, int N, int threads);


/*
Descripción:    Función que obtiene el indice del maximo valor de energia del arreglo 
                de 1D correspondiente al material donde impactaron las particulas.

parametro vector:   Corresponde al vector de energias que representa al material donde
                    impactaron las particulas.

Parametro N: Corresponde a las divisiones del elemento en 1D. Es el largo de este elemento.

Salida:  Devuelve el indice del valor de mayor energia en el array de Energias. 

*/
int get_index_max_energy(float * vector, int N);

#endif