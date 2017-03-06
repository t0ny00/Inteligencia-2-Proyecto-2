# Universidad Simón Bolívar
# Inteligencia Artificial II
# Prof. Ivette Carolina Martínez
# Alumnos:
# Carlos Martinez 11-10584
# Yerson Roa 11-10876
# Antonio Scaramazza 11-10957


Esta carpeta contiene todos los archivos utilizados para la implementacion del segundo proyecto de inteligencia artificial II. A continuacion se describen cada archivo:

NN.py :
	Posee las funciones necesarias para crear la red neuronal, no posee cuerpo main pues este archivo es importado por los programas que realizan las activades 2 y 3 del proyecto.

PerimeterClassification.py:
	Realiza la actividad 2 del proyecto.
	Uso : PerimeterClassification.py <constante_aprendizaje> <numero_neurona_capa_oculta> <num_iteraciones> 											<archivo_entrenamiento>
Iris.py:
	Realiza la actividad 3 del proyecto.
	Uso : Iris.py <constante_aprendizaje> <numero_neurona_capa_oculta> <numero_neuronas_salida> <num_iteraciones> 											<archivo_entrenamiento> <porcentaje_de_corte_de_datos> 
Generator.py:
	Script usado para generar los conjuntos de entrenamiento propios

datos_P2_EM2017_N500.txt,datos_P2_EM2017_N1000,datos_P2_EM2017_N2000
generated_500.txt,generated_1000.txt,generated_2000.txt :
	Conjuntos de entrenamiento para la actividad 2

generated_10000.txt: 
	Conjunto de prueba de la actividad 2

iris.txt:
	Conjunto de entrenamiento original de la actividad 3

iris_2classes.txt:
	Conjunto de entrenamiento de la actividad 3 modificado para identificar si una Iris es o no Setosa


iris_3classes.txt:
	Conjunto de entrenamiento de la actividad 3 modificado para identificar individualmente cada genero de iris