import numpy as np
import sys
import os
import warnings

conf_path = os.getcwd()
sys.path.append(os.path.dirname(conf_path))
sys.path.append(os.path.join(os.path.dirname(conf_path), 'classes'))

from classes.BPLDE import Differential
from classes.Data import Data

warnings.filterwarnings('ignore')

np.set_printoptions(threshold=sys.maxsize)

if __name__ == '__main__':
    data = Data(dataset="heart", target="DEATH_EVENT")
    bplde = Differential(data=data, metric="accuracy", list_exp=["LR", "SVM", "RDC", "KNN", "GNB"],
                         N=50, Gmax=500, LR=0.1, alpha=0.75, mrmr=True, entropy=0.03)
    bplde.init()
