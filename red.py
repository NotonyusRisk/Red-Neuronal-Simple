from scipy import stats
import numpy as np

class capa():
    #numero de neuronas, numero de neuronas de la capa anterior y la funcion de activacion
    def __init__(self, n_neuronas, n_neuronas_ant, funcion_act):
        self.funcion_act = funcion_act
        self.b = np.round(stats.truncnorm.rvs(-1, 1, loc=0, scale=1, size=n_neuronas).reshape(1, n_neuronas), 3)
        self.w = np.round(stats.truncnorm.rvs(-1, 1, loc=0, scale=1, size=n_neuronas * n_neuronas_ant).reshape(n_neuronas_ant, n_neuronas), 3)
        