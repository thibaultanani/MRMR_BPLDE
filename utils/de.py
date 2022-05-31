# Implementation of methods specific to the differential evolution algorithm

from classes import Data
from numpy import ndarray
from scipy import stats
from typing import List
from utils import feature

import numpy as np
import random
import math


def get_cross_proba(muCR: float, scale: float) -> float:
    cross_proba = -1
    while cross_proba > 1 or cross_proba < 0:
        cross_proba = stats.norm.rvs(loc=muCR, scale=scale)
    return cross_proba


def get_pindex(i: int, lst: List, indices: np.array) -> int:
    if i in lst:
        pindex = indices[random.randint(0, len(indices) - 1)]
    else:
        pindex = indices[0]
    return pindex


def get_archive_index(i: int, pindex: int, size: int) -> int:
    while True:
        archive_index = random.randint(0, size - 1)
        if (pindex != archive_index) and (i != archive_index):
            break
    return archive_index


def remove_duplicate(archive: List[List], size: int) -> List[List]:
    uniques = []
    for arr in archive:
        if not any(np.array_equal(arr, unique_arr) for unique_arr in uniques):
            uniques.append(arr)

    archive = uniques

    while len(archive) > size:
        archive.pop(random.randint(0, len(archive) - 1))

    return archive


def update_param(learning_rate: float, muCR: float, SCR: List[float]) -> float:
    try:
        muCR = (1 - learning_rate) * muCR + learning_rate * (sum(SCR)/len(SCR))
    except ZeroDivisionError:
        muCR = (1 - learning_rate) * muCR + learning_rate * 0.5
    return muCR


def evaluation_pop(n_class: int, data: Data, P: ndarray, target: str, metric: str, method: str) -> \
        tuple[list[float], list[float], list[float], list[float], list[float], list[ndarray], list[list[str]]]:
    return feature.fitness(n_class=n_class, d=data, pop=P, target_name=target, metric=metric, method=method)


def evaluation_ind(n_class: int, data: Data, ind: ndarray, target: str, metric: str, method: str) ->\
        tuple[float, float, float, float, float, ndarray, list]:
    return feature.fitness_ind(n_class=n_class, d=data, ind=ind, target_name=target, metric=metric, method=method)


def crossover(n_ind: int, ind: List[bool], mutant: List[bool], cross_proba: float) -> ndarray:
    cross_points = np.random.rand(n_ind) <= cross_proba
    trial = np.where(cross_points, mutant, ind)
    jrand = random.randint(0, n_ind - 1)
    trial[jrand] = mutant[jrand]
    return trial


def mutate(P: ndarray, n_ind: int, i: int, pindex: int, archive_index: int, Xpbest: List[bool],
           P_A: ndarray) -> List[bool]:
    idxs = [idx for idx in range(len(P)) if idx != i and idx != pindex and idx != archive_index]
    selected = np.random.choice(idxs, 2, replace=False)
    Xr1, Xr2 = P[selected]
    Xr2 = P_A[archive_index]
    mutant = []
    for j in range(n_ind):
        # mutant.append(((Xr1[j] ^ Xr2[j]) and Xr1[j]) or (not (Xr1[j] ^ Xr2[j]) and Xpbest[j]))
        if Xr1[j] == Xr2[j]:
            mutant.append(Xpbest[j])
        else:
            mutant.append(Xr1[j])
    return mutant


def get_entropy(pop: ndarray, inds: int, size: int) -> list[float]:
    truth_list = []
    false_list = []
    for i in range(size):
        transpose = [row[i] for row in pop]
        somme = transpose.count(True) / inds
        truth_list.append(somme)
        false_list.append(1 - somme)
    entropy = []
    for j in range(size):
        try:
            log_true = truth_list[j] * math.log(truth_list[j])
        except ValueError:
            log_true = 0.0
        try:
            log_false = false_list[j] * math.log(false_list[j])
        except ValueError:
            log_false = 0.0
        entropy.append(-(log_true + log_false))
    return entropy


