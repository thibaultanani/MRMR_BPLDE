import numpy as np
import sys
import warnings

from classes.BPLDE import Differential
from classes.Data import Data

warnings.filterwarnings('ignore')

np.set_printoptions(threshold=sys.maxsize)


if __name__ == '__main__':
    data = Data(dataset="heart", target="DEATH_EVENT")
    bplde = Differential(data=data, metric="accuracy", list_exp=["LR", "SVM", "RDC", "KNN", "GNB"],
                         N=50, Gmax=500, LR=0.1, alpha=0.75)
    bplde.init()
