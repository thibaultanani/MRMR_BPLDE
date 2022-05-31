# Implementation of methods to manipulate queues

import multiprocessing
from multiprocessing import Queue

from typing import Tuple, List


def init_() -> Tuple[Queue, Queue, Queue, Queue, Queue, Queue, Queue]:
    return multiprocessing.Manager().Queue(), multiprocessing.Manager().Queue(), multiprocessing.Manager().Queue(),\
           multiprocessing.Manager().Queue(), multiprocessing.Manager().Queue(), multiprocessing.Manager().Queue(),\
           multiprocessing.Manager().Queue()


def put_(y2: List[float], folderName: str, scoreMax: int, iteration: int, yTps: List[float], yVars: List[int],
         feature: int, time: float, besties: Queue, names: Queue, names2: Queue, names3: Queue, iters: Queue,
         times: Queue, features: Queue) -> Tuple[Queue, Queue, Queue, Queue, Queue, Queue, Queue]:
    besties.put(y2)
    names.put(folderName + ": " + "{:.2%}".format(scoreMax))
    iters.put(iteration)
    times.put(yTps)
    names2.put(folderName + ": " + str(time))
    features.put(yVars)
    names3.put(folderName + ": " + str(feature))
    return besties, names, iters, times, names2, features, names3


def get_(n_process: int, besties: Queue, names: Queue, names2: Queue, iters: Queue, times: Queue, features: Queue,
         names3: Queue) -> Tuple[list, list, list, list, list, list, list]:
    bestiesLst = []
    namesLst = []
    itersLst = []
    timesLst = []
    names2Lst = []
    featuresLst = []
    names3Lst = []
    for i in range(n_process):
        bestiesLst.append(besties.get())
        namesLst.append(names.get())
        names2Lst.append(names2.get())
        itersLst.append(iters.get())
        timesLst.append(times.get())
        featuresLst.append(features.get())
        names3Lst.append(names3.get())
    return bestiesLst, namesLst, itersLst, timesLst, names2Lst, featuresLst, names3Lst