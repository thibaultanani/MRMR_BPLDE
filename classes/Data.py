from utils import feature

import os
import numpy as np


class Data:
    """
    Represents data and information about a specific dataset

    Parameters
    ----------
    dataset: [str] The name of the file containing the data (.csv or .xlsx)
    target: [str] Target feature for classification/regression
    """
    def __init__(self, dataset: str, target: str):
        self.dataset = dataset
        self.target = target
        # Path to the directory 'in' containing the different datasets as excel/csv
        self.path1 = os.path.dirname(os.getcwd()) + '/in/'
        # Dataset as DataFrame
        self.data = feature.read(filename=(self.path1 + dataset))
        # List of all explanatory features in datasets
        self.cols = self.data.drop([self.target], axis=1).columns
        self.unique, self.count = np.unique(self.data[self.target], return_counts=True)
        # Classes number in the target features
        self.n_class = len(self.unique)
        # Number of explanatory features
        self.D = len(self.cols)

