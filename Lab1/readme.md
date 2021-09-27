# Lab 1 : HPC

Realizado por : 
        - Isidora Abasolo
        - Nicolás Gutiérrez


En este primer laboratorio se verá SIMD- SSE y la transformada de Hough.

## 1 Objetivo
El objetivo de este laboratorio es conocer, usar y apreciar las características de paralelismo a nivel de instrucción de un procesador moderno. Muchas veces desconocemos por completo las capacidades de cómputo SIMD que los procesadores poseen, y rara vez nos preocupamos de explotarlas. En este laboratorio se implementará la Transformada de Hough, para detección de líneas.

## 2 La Transformada de Hough

La transformada de Hough es una operación de procesamiento y análisis de imágenes que se utilizapara encontrar instancias de objetos en una imagen binaria. Las dos aplicaciones más comunes de esta tranformada es para detectar/extraer líneas y círculos. Los objetos que se desean encontrar deben ser representados paramétricamente. Generalmente el proceso completo de extracción de objetos usando la transformada de Hough, incluye etapas de pre y procesamiento. Por ejemplo, si la imagen de entrada es una imagen con niveles de gris, es necesario primero aplicar un filtro de detección de bordes, y luego uno de umbralización (thresholding), y así obtener una imagen binaria (solo con valores cero y 255, por ejemplo). Esta imagen es la entrada a la transformada de Hough.

### 2.1 Detección de líneas

La transformada de Hough es un mapeo del espacio de la imagen al espacio de parámetros del objeto que se desea extraer. Suponga que deseamos detectar todas las líneas rectas en la imagen de entrada.  Una representación paramétrica de una línea es:

```
y = mx + b
```
Luego, el conjunto de píxeles `{(xi,yi),i= 0, 1 ,...,}` que forman una línea con pendiente `m` y desplazamientoben la imagen de entrada, se representa como el punto `(m,b)` en el espacio de parámetros. Note que el espacio parámetro debe ser discretizado, es decir, se debe elegir el muestreo en la dimensión `m` y el muestreo en la dimensión `b`. Sin embargo, existe un problema fundamental con la dimensión `m`, pues todas las líneas rectas perpendiculares, tendrán pendiente infinita, lo cual introduce complejidades a la implementación. Una forma más conveniente de representar paramétricamente una línea es

```
x cosθ + y sinθ = r
```
donde `θ` es el ángulo que forma el vector normal de la recta con el eje `x`, y `r` es la distancia perpendicular de la recta a la línea que pasa por el origen.


Luego, para una imagen de N × N píxeles, se podría elegir la siguiente discretización del espacio paramétrico:

```c
θi = i × ∆θ, con ∆θ = π/M     // donde M representa el número deángulos
```
```c
rj = j × ∆r, con ∆r = N * √2/(2R)    // donde R representa el número de desplazamientos..
```

Un ejemplo concreto de discretización podría ser elegir `M = 180`, y `R=1000`. Luego, el espacio parámetro, o espacio de Hough sería una matriz de M × R entradas.

### 2.2 Detección de líneas

Habiendo elegido el espacio de parámetros y su discretización, el algortimo consiste en recorrer la imagen, pixel a pixel, y para aquellos que pertenecen a alguna línea (aún no sabemos a cuál o a cuáles), incrementar en el espacio parámetro todas las posibles líneas a las cuales podría pertenecer. El siguiente es un pseudo-algoritmo para este proceso:

```c
1 for each pixel (x,y) {
2   if (x,y) is edge {
3       for each theta_i {
4           r_j = x cos(theta_i) + y sin(theta_i)
5           H(theta_i, r_j) = H(theta_i, r_j) + 1
6        }
7   }
8 }
```

Para cada pixel borde de la imagen (pertenece a alguna línea), el algoritmo estima todas las posibles líneas a las cuáles dicho pixel podría pertenecer. Note que todos los píxeles bordes que pertenecen a una línea recta, incrementarán en la misma posición de la matriz H. Luego, al final del procesamiento, `H_(i,j)` contiene el número de píxeles que conforman la línea en `(θi,rj)`. La Figura 3 muestra un ejemplo de imagen de entrada y su transformada de Hough. Una etapa posterior consiste en procesar la matriz H mediante una operación de umbralización, buscando todas las posiciones cuyo valor sea mayor que un cierto valor, lo cual detdectaría todas las líneas de largo (en número de píxeles) igual o mayor al umbral. Estaúltima etapa se realizará usando acceso directo a memoria, y sin uso SIMD SSE.
