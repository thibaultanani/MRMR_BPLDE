from classes.Data import Data
from utils import feature, de, display, queue
from typing import List
from datetime import timedelta
from numpy import ndarray

import os
import time
import psutil
import multiprocessing
import numpy as np
import math
import heapq
import random


class Differential:
    """
    Binary progressive learning differential evolution implementation

    Parameters
    ----------
    data: [Data] Data object containing dataset information
    metric: [string] Choice of metric for optimization [accuracy (default), precision or recall]
    list_exp: [list: string] List of learning methods for experiments running in parallel
              - "LR": logistic regression (default)
              - "SVM": support vector machines
              - "KNN": K-nearest neighbors
              - "RDC": random forest
              - "GNB": gaussian naive bayes
    N: [int] Number of subsets evaluated per generation
    Gmax: [int] Number of generations/iterations for alpha parameter to converge to 1.0
    LR: [float (0.0:1.0)] (learning_rate) Speed at which the crossover probability is going to converge toward 0 or 1
    alpha: [float] Speed at which the number of p decrease for selecting one of p_best indivuals
    mrmr: [bool] Choice to use the mrmr for the initialization of the metaheuristic
    entropy: [float] Population diversity threshold for stopping the algorithm
    """

    def __init__(self, data: Data, metric: str, list_exp: List[str], N: int, Gmax: int, LR: float, alpha: float,
                 mrmr: bool, entropy: float):
        self.Data = data
        self.metric = metric
        self.list_exp = list_exp
        self.N = N
        self.Gmax = Gmax
        self.LR = LR
        self.alpha = alpha
        self.mrmr = mrmr
        if self.mrmr:
            self.mrmr_cols = feature.filtering(data=self.Data.data, target=self.Data.target, k=self.Data.D,
                                               dataset=self.Data.dataset)
        else:
            self.mrmr_cols = None
        self.entropy = entropy
        # Path to the directory 'out' containing the logs and graphics
        self.path2 = os.path.join(os.path.dirname(os.getcwd()), os.path.join('out', self.Data.dataset))
        feature.cleanOut(path=self.path2)

    def write_res(self, folderName: str, probas: List[float], y1: List[float], y2: List[float], colMax: List[str],
                  bestScorePro: List[float], bestAPro: List[float], bestPPro: List[float], bestRPro: List[float],
                  bestFPro: List[float], bestModelPro: List[ndarray], bestScore: float, bestScoreA: float,
                  bestScoreP: float, bestScoreR: float, bestScoreF: float, bestModel: ndarray, bestInd: List[bool],
                  debut: float, out: str, yTps, yVars, method: str) -> None:
        a = os.path.join(os.path.join(self.path2, folderName), 'results.txt')
        f = open(a, "w")
        string = "heuristic: Binary progressive learning differential evolution" + os.linesep + \
                 "method: " + str(method) + os.linesep + \
                 "population: " + str(self.N) + os.linesep + \
                 "generation: " + str(self.Gmax) + os.linesep + \
                 "learning_rate: " + str(self.LR) + os.linesep + \
                 "alpha parameter: " + str(self.alpha) + os.linesep + \
                 "crossover probabilities: " + str(probas) + os.linesep + \
                 "mean: " + str(y1) + os.linesep + "best: " + str(y2) + os.linesep + \
                 "time: " + str(yTps) + os.linesep + \
                 "number of features: " + str(yVars) + os.linesep + \
                 "scores: " + str(bestScorePro) + os.linesep + "accuracies: " + str(bestAPro) + os.linesep + \
                 "precisions: " + str(bestPPro) + os.linesep + "recalls: " + str(bestRPro) + os.linesep + \
                 "F1-scores: " + str(bestFPro) + os.linesep + "models: " + str(bestModelPro) + os.linesep + \
                 "best score: " + str(bestScore) + os.linesep + "best accuracy: " + str(bestScoreA) + \
                 os.linesep + "best precision: " + str(bestScoreP) + os.linesep + "best recall: " + \
                 str(bestScoreR) + os.linesep + "best F1-score: " + str(bestScoreF) + os.linesep + \
                 "best model: " + str(bestModel) + os.linesep + "best subset: " + str(bestInd) + \
                 os.linesep + "columns: " + str(colMax) + os.linesep + \
                 "execution total: " + str(timedelta(seconds=(time.time() - debut))) + os.linesep + \
                 "memory: " + str(psutil.virtual_memory())
        f.write(string)
        f.close()
        a = os.path.join(os.path.join(self.path2, folderName), 'log.txt')
        f = open(a, "w")
        f.write(out)

    def differential_evolution(self, part, process_id, besties, names, iters, times, names2, features, names3):
        debut = time.time()
        print_out = ""
        for mode in part:
            np.random.seed(None)
            folderName = mode.upper() + "_" + str(process_id)
            method = feature.getMethod(mode)
            feature.createDirectory(path=self.path2, folderName=folderName)
            # The axes for the graphics
            x1, y1, y2, yTps, yVars = [], [], [], [], []
            scoreMax, modelMax, indMax, colMax, scoreAMax, scorePMax, scoreRMax, scoreFMax = 0, 0, 0, 0, 0, 0, 0, 0
            # Progression of top performers
            bestScorePro, bestModelPro, bestColsPro, bestIndsPro, bestAPro, bestPPro, bestRPro, bestFPro = \
                [], [], [], [], [], [], [], []
            # Measuring the execution time
            instant = time.time()
            # Parameters initialisation
            muCR, muCRLst, G = 0.5, [], 0
            # Population P initialisation
            if self.mrmr:
                # Try to find best K randomly
                P = feature.create_population_mrmr(inds=self.N * 5, size=self.Data.D, mrmr_cols=self.mrmr_cols,
                                                   data_cols=self.Data.cols)
            else:
                P = feature.create_population(inds=self.N, size=self.Data.D)
            # Archive A initialisation
            A = []
            # Evaluates population
            scores, scoresA, scoresP, scoresR, scoresF, models, cols = \
                de.evaluation_pop(n_class=self.Data.n_class, data=self.Data.data, P=P, target=self.Data.target,
                                  metric=self.metric, method=method)
            if self.mrmr:
                # We take the best mrmr feature subset and add random solutions to create population
                P_tmp = feature.create_population(inds=self.N - 1, size=self.Data.D)
                index = (-np.array(scores)).argsort()[::1][0]
                scores_tmp, scoresA_tmp, scoresP_tmp, scoresR_tmp, scoresF_tmp, models_tmp, cols_tmp = \
                    de.evaluation_pop(n_class=self.Data.n_class, data=self.Data.data, P=P_tmp, target=self.Data.target,
                                      metric=self.metric, method=method)
                scores_tmp.append(scores[index])
                scoresA_tmp.append(scoresA[index])
                scoresP_tmp.append(scoresP[index])
                scoresR_tmp.append(scoresR[index])
                scoresF_tmp.append(scoresF[index])
                models_tmp.append(models[index])
                cols_tmp.append(cols[index])
                P = np.vstack((P_tmp, P[index]))
                scores, scoresA, scoresP, scoresR, scoresF, models, cols = scores_tmp, scoresA_tmp, scoresP_tmp, \
                                                                           scoresR_tmp, scoresF_tmp, models_tmp, \
                                                                           cols_tmp

            bestScore, worstScore, bestModel, bestInd, bestCols, bestScoreA, bestScoreP, bestScoreR, \
            bestScoreF, bestScorePro, bestModelPro, bestIndsPro, bestColsPro, bestAPro, bestPPro, bestRPro, bestFPro = \
                feature.add(scores=scores, models=models, inds=P, cols=cols, scoresA=scoresA, scoresP=scoresP,
                            scoresR=scoresR, scoresF=scoresF, bestScorePro=bestScorePro, bestModelPro=bestModelPro,
                            bestIndsPro=bestIndsPro, bestColsPro=bestColsPro, bestAPro=bestAPro, bestPPro=bestPPro,
                            bestRPro=bestRPro, bestFPro=bestFPro)
            mean_scores = float(np.mean(heapq.nlargest(int(self.N / 2), scores)))
            time_instant = timedelta(seconds=(time.time() - instant))
            time_debut = timedelta(seconds=(time.time() - debut))
            x1, y1, y2, yTps, yVars = \
                feature.add_axis(bestScore=bestScore, meanScore=mean_scores, iteration=G, n_features=len(bestCols),
                                 time_debut=time_debut, x1=x1, y1=y1, y2=y2, yTps=yTps, yVars=yVars)
            # Calculate diversity in population
            entropy = de.get_entropy(pop=P, inds=self.N, size=self.Data.D)
            # Pretty print the results
            print_out = display.pretty_print(print_out=print_out, ID=process_id, method=method, mean=mean_scores,
                                             bestScore=bestScore, numCols=len(bestCols), time_exe=time_instant,
                                             time_total=time_debut, entropy=entropy, iteration=G, p="/") + "\n"
            # Avoid the Archive to be empty
            A.append(bestInd)
            # Main process iteration (generation iteration)
            while True:
                instant = time.time()
                # Successful crossover rates
                sCR = []
                # p parameter calculation
                p = max(1, round(self.N * (1 - (math.sqrt((G / self.Gmax) * self.alpha)))))
                # P' calculation
                mylist = list(range(self.N))
                random.shuffle(mylist)
                half = int(len(mylist) // 2)
                Pprime = mylist[:half]
                # Mutant population creation and evaluation
                for i in range(self.N):
                    indices = (-np.array(scores)).argsort()[:p]
                    # Crossover Rate for i
                    CR = de.get_cross_proba(muCR=muCR, scale=0.1)
                    # Union between the population P and the archive A
                    P_A = np.vstack((P, A))
                    # Choose the value of Xpbest
                    pindex = de.get_pindex(i=i, lst=Pprime, indices=indices)
                    Xpbest = P[pindex]
                    archive_index = de.get_archive_index(i=i, pindex=pindex, size=len(P_A))
                    # Mutant calculation Vi
                    Vi = de.mutate(P=P, n_ind=self.Data.D, i=i, pindex=pindex, archive_index=archive_index,
                                   Xpbest=Xpbest, P_A=P_A)
                    # Trial vector calculation Ui
                    Ui = de.crossover(n_ind=self.Data.D, ind=P[i], mutant=Vi, cross_proba=CR)
                    # Evaluation of the trial vector
                    score_m, accuracy_m, precision_m, recall_m, fscore_m, model_m, col_m = \
                        de.evaluation_ind(n_class=self.Data.n_class, data=self.Data.data, ind=Ui,
                                          target=self.Data.target, metric=self.metric, method=method)
                    # Comparison between Xi and Ui
                    if scores[i] < score_m or ((scores[i] == score_m) and (len(cols[i]) > len(col_m))):
                        # Add the rejected individual in the Archive
                        A.append((P[i]))
                        # Update population
                        P[i], scores[i], scoresA[i], scoresP[i], scoresR[i], scoresF[i], models[i], cols[i] = \
                            Ui, score_m, accuracy_m, precision_m, recall_m, fscore_m, model_m, col_m
                        # Add the successful crossover rates
                        sCR.append(CR)
                        bestScore, worstScore, bestModel, bestInd, bestCols, bestScoreA, bestScoreP, bestScoreR, \
                        bestScoreF, bestScorePro, bestModelPro, bestIndsPro, bestColsPro, bestAPro, bestPPro, \
                        bestRPro, bestFPro = \
                            feature.add(scores=scores, models=models, inds=P, cols=cols, scoresA=scoresA,
                                        scoresP=scoresP, scoresR=scoresR, scoresF=scoresF, bestScorePro=bestScorePro,
                                        bestModelPro=bestModelPro, bestIndsPro=bestIndsPro, bestColsPro=bestColsPro,
                                        bestAPro=bestAPro, bestPPro=bestPPro, bestRPro=bestRPro, bestFPro=bestFPro)
                        A = de.remove_duplicate(archive=A, size=self.N)
                # Update muCR params
                muCR = de.update_param(learning_rate=self.LR, muCR=muCR, SCR=sCR)
                muCRLst.append(muCR)
                G = G + 1
                mean_scores = float(np.mean(heapq.nlargest(int(self.N / 2), scores)))
                time_instant = timedelta(seconds=(time.time() - instant))
                time_debut = timedelta(seconds=(time.time() - debut))
                entropy = de.get_entropy(pop=P, inds=self.N, size=self.Data.D)
                print_out = display.pretty_print(print_out=print_out, ID=process_id, method=method, mean=mean_scores,
                                                 bestScore=bestScore, numCols=len(bestCols),
                                                 time_exe=time_instant, time_total=time_debut, entropy=entropy,
                                                 iteration=G, p=p) + "\n"
                x1, y1, y2, yTps, yVars = \
                    feature.add_axis(bestScore=bestScore, meanScore=mean_scores, iteration=G, n_features=len(bestCols),
                                     time_debut=time_debut, x1=x1, y1=y1, y2=y2, yTps=yTps, yVars=yVars)
                # Create graphics at each generation
                feature.plot(x1=x1, y1=y1, y2=y2, yTps=yTps, yVars=yVars, n_pop=self.N, n_gen=self.Gmax,
                             heuristic="Binary progressive learning differential evolution",
                             folderName=folderName, path=self.path2, bestScore=bestScore,
                             mean_scores=mean_scores, time_total=time_debut.total_seconds(), metric=self.metric)
                # Update which individual is the best
                if bestScore > scoreMax:
                    scoreMax, modelMax, indMax, colMax, scoreAMax, scorePMax, scoreRMax, scoreFMax = \
                        bestScore, bestModel, bestInd, bestCols, bestScoreA, bestScoreP, bestScoreR, bestScoreF
                # Write important information to file
                self.write_res(folderName=folderName, probas=muCRLst, y1=y1, y2=y2, colMax=colMax,
                               bestScorePro=bestScorePro, bestAPro=bestAPro, bestPPro=bestPPro, bestRPro=bestRPro,
                               bestFPro=bestFPro, bestModelPro=bestModelPro, bestScore=bestScore, bestScoreA=scoreAMax,
                               bestScoreP=scorePMax, bestScoreR=scoreRMax, bestScoreF=scoreFMax, bestModel=modelMax,
                               bestInd=indMax, debut=debut, out=print_out, yTps=yTps, yVars=yVars, method=method)
                # No need to continue when all individuals are the same
                if sum(entropy) / len(entropy) < self.entropy:
                    break
            besties, names, iters, times, names2, features, names3 = \
                queue.put_(y2=y2, folderName=folderName, scoreMax=scoreMax, iteration=G, yTps=yTps,
                           time=time_debut.total_seconds(), besties=besties, names=names, names2=names2, iters=iters,
                           times=times, features=features, names3=names3, yVars=yVars, feature=len(bestCols))

    def init(self):
        print("####################################################")
        print("#BINARY PROGRESSIVE LEARNING DIFFERENTIAL EVOLUTION#")
        print("####################################################")
        print()
        besties, names, iters, times, names2, features, names3 = queue.init_()
        mods = self.list_exp
        n = len(mods)
        mods = [mods[i::n] for i in range(n)]
        arglist = []
        process_id = 1
        for part in mods:
            arglist.append((part, process_id, besties, names, iters, times, names2, features, names3))
            process_id = process_id + 1
        pool = multiprocessing.Pool(processes=len(mods))
        pool.starmap(self.differential_evolution, arglist)
        pool.close()
        bestiesLst, namesLst, itersLst, timesLst, names2Lst, featuresLst, names3Lst = \
            queue.get_(n_process=len(mods), besties=besties, names=names, names2=names2, iters=iters, times=times,
                       features=features, names3=names3)
        pool.join()
        return feature.res(heuristic="Binary progressive learning differential evolution", besties=bestiesLst,
                           names=namesLst, times=timesLst, names2=names2Lst, features=featuresLst, names3=names3Lst,
                           path=self.path2, dataset=self.Data.dataset)
