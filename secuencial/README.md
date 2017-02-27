# Detalles del código secuencial

La fase de etiquetado utiliza 3 matrices del tamaño de la imagen con una fila
y una columna extra al principio y final de la matriz (halo):
1. Matriz con los índices de color de cada pixel (en los posiciones del halo
  se coloca el valor 0)
2. Matriz de etiquetas
3. Copia de la matriz de etiquetas


La matriz de etiquetas se inicializa de la siguiente forma:
- Pixels con índice 0: Se etiquetan con el valor -1
- Pixels con índices entre 1 y 15: Se etiquetan con la posición   del elemento
  de la matriz: _i*NUMCOLS+j_

La fase de etiquetado es un bucle en el que en cada paso:
- Se copian los datos de la matriz de etiquetas en la matriz auxiliar de copia
- Se recorre la matriz de etiquetas (esquivando el halo) actualizando cada
  posición con la etiqueta de menor valor entre la suya propia o una de las de
  sus cuatro vecinos que tenga el mismo índice de color.

Se repite el proceso hasta que en un iteración ninguna celda de la matriz
cambia de etiqueta. En ese momento cada masa de pixeles contiguos del mismo
índice tiene la misma etiqueta, que corresponde con la posición lineal del
primer pixel del objeto (pixel más arriba y a la izquierda). Finalmente se
recorre la matriz de etiquetas contando el número de celdas que tienen como
etiqueta su propia posición lineal. El resultado es el número de objetos
diferentes.

(c) 2017, Grupo Trasgo, Universidad de Valladolid
