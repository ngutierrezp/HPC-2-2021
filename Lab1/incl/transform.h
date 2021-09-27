#ifndef TRANSFORM_H
#define TRANSFORM_H
/*
Descripción:    Función que se encarga de realizar la transformada de Hough dada una imagen en forma de matriz.
                El proceso se realiza tomando en consideración lo siguiente:
                
                    1 for each pixel (x,y) {
                    2   if (x,y) is edge {
                    3       for each theta_i {
                    4           r_j = x cos(theta_i) + y sin(theta_i)
                    5           H(theta_i, r_j) = H(theta_i, r_j) + 1
                    6        }
                    7   }
                    8 }
                La Parametrización utilizada en este caso es : 
                    θi = i × ∆θ, con ∆θ = π/M
                    rj = j × ∆r, con ∆r = √(N*N + M*M)/(R)

parametro matriz: Es una matriz de enteros que representa la imagen .raw leida.

parametro N: Es un entero que representa el tamño de la imagen en X

parametro M: Es un entero que representa el tamño de la imagen en Y

parametro T: Es un entero que representa la cantidad de angulos.

parametro R: Es un entero que representa la cantidad de distancias.

parametro offset: Es un entero que representa la cantidad de espacio para ajustar en las distancias.

Salida:  Genera la matriz de hough de tamaño TxR.
*/
int **hough_transform(int **matriz, int N, int M, int T, int R,int offset);



/*
Descripción:    Función que toma los valores de la matriz de hough y si son mayores a un umbral
                lleva el valor escogido a un maximo de 255. En otras palabras todo valor mayor 
                al umbral pasa a tomar el valor 255 y los menores al umbral toman valor de 0.

parametro matriz: Es una matriz de hough.

parametro T: Es un entero que representa la cantidad de angulos.

parametro R: Es un entero que representa la cantidad de distancias.

parametro U: Es un entero que representa el umbral.

Salida:  Genera la matriz de hough de tamaño TxR.
*/
void umbralization(int **matriz, int T, int R, int U);

/*
Descripción:    Función que se encarga de realizar la transformada de Hough dada una imagen en forma de matriz.
                El proceso se realiza tomando en consideración lo siguiente:
                
                    1 for each pixel (x,y) {
                    2   if (x,y) is edge {
                    3       for each theta_i {
                    4           r_j = x cos(theta_i) + y sin(theta_i)
                    5           H(theta_i, r_j) = H(theta_i, r_j) + 1
                    6        }
                    7   }
                    8 }
                La Parametrización utilizada en este caso es : 
                    θi = i × ∆θ, con ∆θ = π/M
                    rj = j × ∆r, con ∆r = √(N*N + M*M)/(R)

                El proceso se realiza de forma paralela ya que toma 4 angulos y los procesa de forma
                simultanea.

parametro matriz: Es una matriz de enteros que representa la imagen .raw leida.

parametro N: Es un entero que representa el tamño de la imagen en X

parametro M: Es un entero que representa el tamño de la imagen en Y

parametro T: Es un entero que representa la cantidad de angulos.

parametro R: Es un entero que representa la cantidad de distancias.

parametro offset: Es un entero que representa la cantidad de espacio para ajustar en las distancias.

Salida:  Genera la matriz de hough de tamaño TxR.
*/
int **SIMD_hough_transform(int **matriz, int N, int M, int T, int R, int offset);


#endif