import os
import numpy as np


pesos_tamanhos = [640,640,640,640,640,640,640,32]
  

if not os.path.isdir("./input"):

    
    os.makedirs("./input")




def criar_matriz(x,y, file_name, n_peso=None):
    nova_matriz = np.random.uniform(-1,1,x*y)
    
    nova_matriz = nova_matriz.reshape(x,y)
    nova_matriz = nova_matriz.round( 4);

    f = open(file_name,"w")
    f.write("{} {}".format(x,y))
    if n_peso != None:
        f.write(" {}".format(n_peso))
    f.write("\n")
    for i in range(x):
        f.write(" ".join(  list(  map(  str, nova_matriz[i])  )  ) + "\n")




criar_matriz(pesos_tamanhos[0], pesos_tamanhos[1], "./input/entrada",n_peso=len(pesos_tamanhos)-2)
row = pesos_tamanhos[1]
for interador, i in enumerate(pesos_tamanhos[2:]):
    
    # print(interador,i)
    criar_matriz(row, i, "./input/peso-{}".format(interador))
    row = i
# criar_matriz(10,10, "./input/peso-{}".format(1))