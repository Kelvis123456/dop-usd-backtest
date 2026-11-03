# Decisiones

Cada entrada dice qué elegí, qué descarté y por qué.

## 2026-11-03 · El naive es el baseline y todo se mide contra él

**Elegí:** repetir el último valor como vara de medir, y reportar cada modelo como diferencia de error contra él, con intervalo.

**Descarté:** reportar el MAE de cada modelo por separado. Un 0.19 suena bien hasta que ves que el naive saca 0.18. Lo que importa es la diferencia, y si su intervalo cruza el cero.

## 2026-11-03 · Cómo se evita mirar el futuro

El modelo recibe `values[: origen + 1]` y nada más. Lo comprobé de tres formas en los tests: un modelo espía que guarda lo que recibe y se compara con el prefijo de la serie; cambiar todo lo posterior al origen no cambia el pronóstico; y la prueba de que el test puede fallar. Metí a propósito `values[: origen + 2]` y ahora fallan 5 tests, así que sé que el arnés detecta una fuga de un solo día.

La primera versión no era tan buena: el test de "cambiar el futuro" cortaba en el índice 400, que no es un origen (son 364, 371, ..., 399, 406), y por eso no detectaba esa fuga; los 3 fallos de entonces eran el espía y dos casos que caían por casualidad. Una revisión posterior lo encontró y lo corté en un origen real. También encontró una fuga que sí existía: el único día que falta en los datos es un origen, y interpolarlo en línea recta le mostraba al modelo el día siguiente. Ahora se repite el valor anterior.

## 2026-11-03 · No limpiar la serie

La serie tiene días sueltos con valores raros. Pensé en quitarlos con una mediana centrada de 5 días, pero esa mediana usa los dos días siguientes: en el origen `t` el modelo vería `t+1` y `t+2` limpiando. Un filtro que solo mire hacia atrás no puede detectar el último día, que es justo el que importa en el origen. Dejé los datos crudos y agregué `median5` como un modelo más, que es lo que haría alguien con esos datos sin poder limpiarlos. Salió peor que el naive; creo que el ruido es raro y promediar cinco días atrasa más de lo que ayuda, pero no lo probé aparte.

## 2026-11-03 · Intervalos con bootstrap por bloques

Con un origen cada 7 días y horizonte de 30, el error del origen `t` y el de `t+7` comparten 23 de sus 30 días. Un bootstrap común trata cada origen como independiente y da intervalos demasiado estrechos. Remuestreo bloques circulares de `ceil(h / 7)` orígenes. Con 83 orígenes los intervalos siguen siendo anchos, y los dejo anchos en vez de subir el número de orígenes con un paso de 1 día, que solo duplicaría el solape.

## 2026-11-03 · statsmodels y no un modelo más grande

ETS y ARIMA porque son los que cualquiera esperaría ver y los que mejor se explican. Descarté gradient boosting sobre rezagos: con 600 puntos de una serie casi sin estructura, lo más probable es que ajuste ruido, y un resultado "mejor que el naive" ahí habría que desconfiarlo más que uno peor.
