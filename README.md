# tfg
El programa hace lo siguiente:
1)Toma dos capturas consecutivas.

2)Realiza la búsqueda de puntos característicos y calcula sus descriptores.

3)Realiza el matching entre los puntos característicos de dos frames consecutivos.

4)Calcula la matriz esencial.

5)Obtiene la rotación y traslación entre dos frames consecutivos a través de la matriz esencial

6)si se trata de las dos primeras capturas se actualiza la posición de la cámara y se calculan por triangulación
los primeros puntos 3d de la nube.Si no se procede a obtener unos puntos 3d provisionales por triangulación
que se utilizarán para el cómputo de la escala relativa.

7)Si se trata de las dos primeras imágenes actualiza la nube de puntos 3d.
Si se trata de imágenes sucesivas realiza el cálculo de la escala relativa. Para ello busca entre 3 frames consecutivos
(Ik-2,Ik-1,Ik) puntos 3d correspondientes y calcula el factor de escala como el cociente entre la norma euclídea del
vector formado por dos puntos 3d obtenidos a través de los instantes Ik-2 e Ik-1 y la norma euclídea del vector de sus 
correspondientes puntos 3d entre los frames Ik-1 e Ik.
Para que sea más robusto y teniendo en cuenta que aún pueden quedar outliers,se calcula un vector de escalas a través
de distintos pares de puntos 3d correspondientes entre los 3 frames mencionados y se considera que la escala a aplicar
al vector de traslación es la mediana de dicho vector.

8)una vez tenemos la escala relativa,se reescala el vector de traslación,se actualiza la posición de la cámara,se
calculan nuevamente los puntos 3d por triangulación y se actualiza la nube de puntos 3d

9)el proceso se repite para todo el conjunto de imágenes y cuando estan todas procesadas se realiza una representación
gráfica a través de PCL de la nube de puntos y de las distintas posiciones de la cámara.

El programa aún no funciona,creo que es debido a que la distancia entre imágenes es muy pequeña,y obtengo malas
triangulaciones.
He probado a aumentar la distancia entre capturas tomando solo algunas de las imágenes y aun así no obtengo resultados 
coherentes.

La aplicación debe estar en el mismo directorio que las imágenes del DATASET 
