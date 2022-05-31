# Implementation of methods specific to the feature selection problem
from datetime import timedelta

import numpy as np
import shutil
import pandas as pd
import difflib
import os
import random
import matplotlib.pyplot as plt
import openpyxl
import json

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn import model_selection
from sklearn.utils import class_weight

from typing import Union, Sequence, Tuple, List
from numpy import ndarray
from mrmr import mrmr_classif


def cleanOut(path: str) -> None:
    # Clears the record of previous experiments on the same dataset
    final = path
    try:
        shutil.rmtree(final)
    except FileNotFoundError:
        pass


def read(filename: str) -> pd.DataFrame:
    # Returns the data contained in a xlsx/csv file as Dataframe
    try:
        data = pd.read_excel(filename + '.xlsx', index_col=None, engine='openpyxl')
    except FileNotFoundError:
        data = pd.read_csv(filename + '.csv', index_col=None, sep=',')
    return data


def createDirectory(path: str, folderName: str) -> None:
    # Create the folder to save the results
    final = os.path.join(path, folderName)
    if os.path.exists(final):
        shutil.rmtree(final)
    os.makedirs(final)


def getMethod(method: str) -> Union[Sequence, str]:
    # Returns the learning method closest to the string passed as an argument
    words = ['lr', 'svm', 'knn', 'rdc', 'gnb']
    try:
        return difflib.get_close_matches(method.lower(), words)[0]
    except IndexError:
        return 'lr'


def create_population(inds: int, size: int) -> ndarray:
    # Initialise the population
    pop = np.zeros((inds, size), dtype=bool)
    for i in range(inds):
        pop[i, 0:random.randint(0, size)] = True
        np.random.shuffle(pop[i])
    return pop


def create_population_mrmr(inds: int, size: int, mrmr_cols: List[str], data_cols: List[str]) -> ndarray:
    # Initialise the population with mrmr
    pop = np.zeros((inds, size), dtype=bool)
    for i in range(inds):
        k = random.randint(1, size)
        for j in range(len(data_cols)):
            if data_cols[j] in mrmr_cols[0:k]:
                pop[i][j] = True
            else:
                pop[i][j] = False
    return pop


def preparation(data: pd.DataFrame, ind: ndarray, target: str) -> Tuple[pd.DataFrame, List[str]]:
    # Selects columns based on the value of an individual
    copy = data.copy()
    copy_target = copy[target]
    copy = copy.drop([target], axis=1)
    cols = copy.columns
    cols_selection = []
    for c in range(len(cols)):
        if ind[c]:
            cols_selection.append(cols[c])
    copy = copy[cols_selection]
    copy[target] = copy_target
    return copy, cols_selection


def cross_validation(nfold: int, X: list, y: list, model, matrix: ndarray) -> Tuple[ndarray, list, list]:
    # Cross-validation to ensure that learning is representative of the whole dataset
    k = model_selection.StratifiedKFold(nfold)
    y_test_lst = []
    y_pred_lst = []
    # Allows the data to be separated into k distributions. For each distribution, a learning process is performed
    for train_index, test_index in k.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        sample = class_weight.compute_sample_weight('balanced', y_train)
        try:
            model.fit(X_train, y_train, sample_weight=sample)
        except TypeError:
            model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # Sum of the confusion matrices for each of the distributions
        matrix = matrix + confusion_matrix(y_test, y_pred)
        y_test_lst.extend(y_test)
        y_pred_lst.extend(y_pred)
    return matrix, y_test_lst, y_pred_lst


def learning(n_class: int, data: pd.DataFrame, target: str, method: str) -> Tuple[float, float, float, float, ndarray]:
    # Performs learning according to the chosen method
    X = data.drop([target], axis=1).values
    y = data[target].values
    matrix = np.zeros((n_class, n_class), dtype=int)
    if method == 'svm':
        model = LinearSVC(random_state=1)
    elif method == 'knn':
        model = KNeighborsClassifier(n_neighbors=5)
    elif method == 'rdc':
        model = RandomForestClassifier(n_estimators=30, random_state=1)
    elif method == 'dtc':
        model = DecisionTreeClassifier(random_state=1)
    elif method == 'etc':
        model = ExtraTreesClassifier(class_weight='balanced', random_state=1)
    elif method == 'lda':
        model = LinearDiscriminantAnalysis()
    elif method == 'gnb':
        model = GaussianNB()
    elif method == 'rrc':
        model = RidgeClassifier(class_weight='balanced')
    else:
        model = LogisticRegression(solver='liblinear', C=10.0)
    matrix, y_test, y_pred = cross_validation(nfold=5, X=X, y=y, model=model, matrix=matrix)
    return accuracy_score(y_true=y_test, y_pred=y_pred),\
           precision_score(y_true=y_test, y_pred=y_pred, average="macro"),\
           recall_score(y_true=y_test, y_pred=y_pred, average="macro"), \
           f1_score(y_true=y_test, y_pred=y_pred, average="macro"), matrix


def fitness_ind(n_class: int, d: pd.DataFrame, ind: ndarray, target_name: str, metric: str, method: str) ->\
        Tuple[float, float, float, float, float, ndarray, list]:
    # Process of calculating the fitness for a single individual/subset
    if not any(ind):
        ind[random.randint(0, len(ind) - 1)] = True
    data, cols = preparation(data=d, ind=ind, target=target_name)
    accuracy, precision, recall, f_score, matrix = learning(n_class=n_class, data=data, target=target_name,
                                                            method=method)
    metric = metric.lower()
    if metric == 'accuracy':
        score = accuracy
    elif metric == 'recall':
        score = recall
    elif metric == 'precision':
        score = precision
    else:
        score = accuracy
    return score, accuracy, precision, recall, f_score, matrix, cols


def fitness(n_class: int, d: pd.DataFrame, pop: ndarray, target_name: str, metric: str, method: str) -> \
        Tuple[List[float], List[float], List[float], List[float], List[float], List[ndarray], List[List[str]]]:
    # Process of calculating the fitness for each individual of a population
    score_list, accuracy_list, precision_list, recall_list, fscore_list, col_list, matrix_list =\
        [], [], [], [], [], [], []
    metric = metric.lower()
    for ind in pop:
        if not any(ind):
            ind[random.randint(0, len(ind) - 1)] = True
        data, cols = preparation(data=d, ind=ind, target=target_name)
        accuracy, precision, recall, f_score, matrix = learning(n_class=n_class, data=data, target=target_name,
                                                                method=method)
        if metric == 'accuracy':
            score_list.append(accuracy)
        elif metric == 'recall':
            score_list.append(recall)
        elif metric == 'precision':
            score_list.append(precision)
        else:
            score_list.append(accuracy)
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        fscore_list.append(f_score)
        col_list.append(cols)
        matrix_list.append(matrix)
    return score_list, accuracy_list, precision_list, recall_list, fscore_list, matrix_list, col_list


def add(scores: List[float], models: List[ndarray], inds: ndarray, cols: List[List[str]], scoresA: List[float],
        scoresP: List[float], scoresR: List[float], scoresF: List[float], bestScorePro: List[float],
        bestModelPro: List[ndarray], bestIndsPro: List[List[bool]], bestColsPro: List[List[str]], bestAPro: List[float],
        bestPPro: List[float], bestRPro: List[float], bestFPro: List[float]) ->\
        Tuple[float, float, ndarray, List[bool], List[str], float, float, float, float, List[float], List[ndarray],
              List[List[bool]], List[List[str]], List[float], List[float], List[float], List[float]]:
    argmax = np.argmax(scores)
    argmin = np.argmin(scores)
    bestScore = scores[argmax]
    worstScore = scores[argmin]
    bestModel = models[argmax]
    bestInd = inds[argmax]
    bestCols = cols[argmax]
    bestScoreA = scoresA[argmax]
    bestScoreP = scoresP[argmax]
    bestScoreR = scoresR[argmax]
    bestScoreF = scoresF[argmax]
    bestScorePro.append(bestScore)
    bestModelPro.append(bestModel)
    bestIndsPro.append(bestInd)
    bestColsPro.append(bestCols)
    bestAPro.append(bestScoreA)
    bestPPro.append(bestScoreP)
    bestRPro.append(bestScoreR)
    bestFPro.append(bestScoreF)
    return bestScore, worstScore, bestModel, bestInd, bestCols, bestScoreA, bestScoreP, bestScoreR, bestScoreF, \
           bestScorePro, bestModelPro, bestIndsPro, bestColsPro, bestAPro, bestPPro, bestRPro, bestFPro


def add_axis(bestScore: float, meanScore: float, iteration: int, time_debut: timedelta, n_features: int, x1: List[int],
             y1: List[float], y2: List[float], yTps: List[float], yVars: List[int]) ->\
        Tuple[List[int], List[float], List[float], List[float], List[int]]:
    x1.append(iteration)
    y1.append(meanScore)
    y2.append(bestScore)
    yTps.append(time_debut.total_seconds())
    yVars.append(n_features)
    return x1, y1, y2, yTps, yVars


def plot_feature(x1: List[int], y1: List[float], y2: List[float], yTps: List[float], yVars: List[int],
                 n_pop: int, n_gen: int, heuristic: str, folderName: str, path: str, bestScore: float,
                 mean_scores: float, time_total: float, metric: str) -> None:
    fig, ax = plt.subplots()
    ax.plot(x1, y1)
    ax.plot(x1, y2)
    ax.set_title("Score evolution per generation (" + folderName + ")" + "\n" + heuristic + "\n")
    ax.set_xlabel("generation")
    ax.set_ylabel(metric)
    ax.grid()
    ax.legend(labels=["Mean of the top " + str(int(n_pop / 2)) + ": " + "{:.2%}".format(mean_scores),
                      "Best: " + "{:.2%}".format(bestScore)],
              loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
    a = os.path.join(os.path.join(path, folderName), 'plotScore_' + str(n_gen) + '.png')
    b = os.path.join(os.getcwd(), a)
    fig.savefig(os.path.abspath(b), bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots()
    ax2.plot(x1, yVars)
    ax2.set_title("Number of features per generation  (" + folderName + ")" + "\n" + heuristic + "\n")
    ax2.set_xlabel("generation")
    ax2.set_ylabel("number of features")
    ax2.grid()
    ax2.legend(labels=["Best: " + str(yVars[len(yVars)-1])],
               loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
    a = os.path.join(os.path.join(path, folderName), 'plotFeatures_' + str(n_gen) + '.png')
    b = os.path.join(os.getcwd(), a)
    fig2.savefig(os.path.abspath(b), bbox_inches="tight")
    plt.close(fig2)

    fig3, ax3 = plt.subplots()
    ax3.plot(x1, yTps)
    ax3.set_title("Time (seconds) per generation (" + folderName + ")" + "\n" + heuristic + "\n")
    ax3.set_xlabel("generation")
    ax3.set_ylabel("Time (seconds)")
    ax3.grid()
    ax3.legend(labels=["Total total: " + "{:0f}".format(time_total)],
               loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
    a = os.path.join(os.path.join(path, folderName), 'plotTime_' + str(n_gen) + '.png')
    b = os.path.join(os.getcwd(), a)
    fig3.savefig(os.path.abspath(b), bbox_inches="tight")
    plt.close(fig3)


def res(heuristic: str, besties: list, names: list, times: list, names2: list, features: list,
        names3: list, path: str, dataset: str) -> None:
    besties = np.array(besties)
    names = np.array(names)
    times = np.array(times)
    names2 = np.array(names2)
    features = np.array(features)
    names3 = np.array(names3)
    indices = names.argsort()
    besties = besties[indices].tolist()
    names = names[indices].tolist()
    times = times[indices].tolist()
    names2 = names2[indices].tolist()
    features = features[indices].tolist()
    names3 = names3[indices].tolist()

    folderName = "Total_" + dataset
    createDirectory(path, folderName)
    fig, ax = plt.subplots()
    i = 0
    for val in besties:
        ax.plot(list(range(0, len(val))), val)
        i = i + 1
    ax.set_title("Score evolution per generation" + "\n" + heuristic + "\n" + dataset)
    ax.set_xlabel("generation")
    ax.set_ylabel("score")
    ax.grid()
    ax.legend(labels=names,
              loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
    a = os.path.join(os.path.join(path, folderName), 'plot_' + '.png')
    b = os.path.join(os.getcwd(), a)
    fig.savefig(os.path.abspath(b), bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots()
    i = 0
    for val in times:
        ax2.plot(list(range(0, len(val))), val)
        i = i + 1
    ax2.set_title("Time (seconds) per generation" + "\n" + heuristic + "\n" + dataset)
    ax2.set_xlabel("generation")
    ax2.set_ylabel("Time (seconds)")
    ax2.grid()
    ax2.legend(labels=names2,
               loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
    a = os.path.join(os.path.join(path, folderName), 'plotTime_' + '.png')
    b = os.path.join(os.getcwd(), a)
    fig2.savefig(os.path.abspath(b), bbox_inches="tight")
    plt.close(fig2)

    fig3, ax3 = plt.subplots()
    i = 0
    for val in features:
        ax3.plot(list(range(0, len(val))), val)
        i = i + 1
    ax3.set_title("Number of features per generation" + "\n" + heuristic + "\n" + dataset)
    ax3.set_xlabel("generation")
    ax3.set_ylabel("Number of features")
    ax3.grid()
    ax3.legend(labels=names3,
               loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
    a = os.path.join(os.path.join(path, folderName), 'plotFeatures_' + '.png')
    b = os.path.join(os.getcwd(), a)
    fig3.savefig(os.path.abspath(b), bbox_inches="tight")
    plt.close(fig3)


def filtering(data: pd.DataFrame, target: str, k: int, dataset: str):
    X = data.drop([target], axis=1)
    y = data[target]

    try:
        # If mrmr selection is already saved
        file = open(os.path.join(os.path.join(os.path.dirname(os.getcwd()), "mrmr_saves"),
                                 dataset + '_' + target + '.txt'), 'r')
        lines = file.readlines()
        sorted_features = []
        for l in lines:
            sorted_features = l.replace("\n", "").replace("\'", "\"")
            sorted_features = json.loads(sorted_features)
    except FileNotFoundError:
        # Otherwise, we calculate mrmr scores
        sorted_features = mrmr_classif(X=X, y=y, K=k)
        try:
            file = open(os.path.join(os.path.join(os.path.dirname(os.getcwd()), "mrmr_saves"),
                                     dataset + '_' + target + '.txt'), 'w')
        except FileNotFoundError:
            createDirectory(path=os.path.dirname(os.getcwd()), folderName="mrmr_saves")
            file = open(os.path.join(os.path.join(os.path.dirname(os.getcwd()), "mrmr_saves"),
                                     dataset + '_' + target + '.txt'), 'w')
        file.write(str(sorted_features))
    X = data[sorted_features].values
    y = data[target]
    cols = sorted_features

    y = y.values

    try:
        cols = cols.tolist()
    except AttributeError:
        pass

    return X, y, cols

