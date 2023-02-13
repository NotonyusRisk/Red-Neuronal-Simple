import numpy as np
import math
import matplotlib.pyplot as plt

sigmoid = (lambda x:1 / (1 + np.exp(-x)), lambda x:x * (1 - x))

rango = np.linspace(-10,10).reshape([50,1])
datos_sigmoide = sigmoid[0](rango)
datos_sigmoide_derivada = sigmoid[1](rango)

#Cremos los graficos
fig, axes = plt.subplots(Nrows=1, ncols=2, figsize =(15, 5))
axes[0].plot(rango, datos_sigmoide)
axes[1].plot(rango, datos_sigmoide_derivada)
fig.tight_layout()

#funcion relu
def derivada_relu(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

relu = (
    lambda x: x * (x > 0),
    lambda x:derivada_relu(x)
)

datos_relu = relu[0](rango)
datos_relu_derivada = relu[1](rango)

rango = np.linspace(-10, 10).reshape([50,1])

plt.cla()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (15,5))
axes[0].plot(rango, datos_relu[:,0])
axes[1].plot(rango, datos_relu_derivada[:,0])
plt.show()