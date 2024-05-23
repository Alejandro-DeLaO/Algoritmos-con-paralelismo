# Primero instalamos las librerias
#! pip install numpy
#! pip install mpi4py


#Luego seleccionamos las librerias numpy y mpi4py
import numpy as np
from mpi4py import MPI

#Creamos una lista de 10,000 numeros aleatorios para el algoritmo sin paralelismo
lista = np.random.randint(1, 101, 10000)

#Este es el algoritmo sin paralelizar
def selectionSort(arr):
  n = len(arr)
  for i in range(n):
    min_index = i
    for j in range (i + 1, n):
      if arr[j] < arr[min_index]:
        min_index = j
    arr[i], arr[min_index] = arr[min_index], arr[i]

#Ejecutamos el algoritmo con la lista anteriormente creada

mi_lista = (lista)
selectionSort(mi_lista)
print(mi_lista)


#Ahora comenzamos con el paralelismo y agregamos el comunicador de procesos y el identificador de rangos.
if __name__ == "__main__":
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()

#Ahora agregamos al proceso de rango 0 una lista de 10,000 numeros.
if rank == 0:
  data = np.random.randint(1, 101, 10000)
else:
  data = None

#Ahora dividimos esa lista entre todos los procesos y cada proceso lo almacena en la variable llamada local_data
local_data = np.empty(10000, dtype=int)
comm.Scatter(data, local_data, root=0)

#En esta parte hacemos que cada proceso ejecute el algoritmo por seleccion
selectionSort(local_data)

#Ahora hacemos que cada proceso mande las sublistas ordenadas al proceso de rango 0
new_data = comm.gather(local_data, root = 0)

#Finalmente cuando ya este todo ordenado en el proceso de rango 0 se imprime la lista ordenada.
if rank == 0:
  print(f"Maestro recopilo: {new_data}")