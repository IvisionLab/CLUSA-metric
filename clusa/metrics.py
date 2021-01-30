#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats
import itertools
import warnings
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from scipy import stats
from scipy.spatial.distance import pdist

from itertools import product

def binarize_dataset(dataset):
    dataset = label_filter(dataset, minimum = 2)
    mydata = dataset
    new_dataset = np.concatenate(list(map(lambda j: label_filter(np.array(list(map(lambda i: (mydata[j] > i).astype(int),
                                                                     set(mydata[j])))), minimum = 2),
                               range(mydata.shape[0]))))

    weights = np.array(list(map(lambda g: np.bincount(g)[0]/float(len(g)), new_dataset)))
    ordering = np.argsort(weights)
    new_dataset = new_dataset[ordering,:]
    return new_dataset

def fmeasure_consistency(dataset):
    try:
        dm = pdist(dataset, lambda u, v: f1_score(u, v, average='binary'))
        return np.nanmean(dm)
    except:
        dm = pdist(dataset, lambda u, v: f1_score(u, v, average='micro'))
        return np.nanmean(dm)

def cronbach_standard(dataset):
    N = dataset.shape[0]
    A = list(product(range(N), range(N)))
    A = list(filter(lambda a: a[1] > a[0], A))
    corr = np.array(list(map(lambda b: pearsonr(dataset[b[0],:], dataset[b[1],:])[0], A)))
    r = np.nanmean(corr)
    ca=(N*r)/(1+(N-1)*r)
    return ca

def cronbach_alpha(dataset):
    N = dataset.shape[1]
    ca=(N/float(N-1))*(1-np.sum(pow(np.std(dataset, axis=0),2))/pow(np.std(dataset.sum(axis=1)),2))
    return ca


class Evaluation:
    train = None
    test = None
    prefix = ''
    ct = 1

    def __init__(self, train=None, test=None):
        self.train = train
        self.test = test

    def set_expected_and_estimated(self, expected, estimated):
        self.train = expected
        self.test = estimated
        return self
    
    def set_prefix(self, prefix):
        self.prefix = prefix
        return self
    
    def run(self, parameter=None):
        uniques = np.unique(self.train)
        pack = list(itertools.product(range(self.train.shape[0]),range(self.test.shape[0])))
        pack = list(zip(*pack))
        scores = np.array(list(map(lambda x,y: self.distance(x,y, uniques), pack[0], pack[1])))
        return (scores, scores.mean())

    def distance(self, idx_train, idx_test, parameter=None):
        raise NotImplementedError("Please Implement this method")


class F1_Score(Evaluation):
    def distance(self, idx_train, idx_test, parameter=None):
        try:
            score = f1_score(self.train[idx_train], self.test[idx_test], labels=parameter, average='binary')
            return score
        except:
            score = f1_score(self.train[idx_train], self.test[idx_test], labels=parameter, average='micro')
            return score
        return scores

    def run(self, parameter=None):
        uniques = np.concatenate((np.unique(self.train),np.unique(self.test)))
        pack = list(itertools.product(range(self.train.shape[0]),range(self.test.shape[0])))
        pack = list(zip(*pack))
        scores = np.array(list(map(lambda x,y: self.distance(x,y, uniques), pack[0], pack[1])))
        return scores
    
class Kendall(Evaluation):
    def distance(self, idx_train, idx_test, parameter=None):
        tau, p_value = stats.kendalltau(self.train[idx_train], self.test[idx_test])
        return tau

    def run(self, parameter=None):
        uniques = np.unique(self.train)
        pack = list(itertools.product(range(self.train.shape[0]),range(self.test.shape[0])))
        pack = list(zip(*pack))
        scores = np.array(list(map(lambda x,y: self.distance(x,y, uniques), pack[0], pack[1])))
        return scores

class Spearman(Evaluation):
    def distance(self, idx_train, idx_test, parameter=None):
        tau, p_value = stats.spearmanr(self.train[idx_train], self.test[idx_test])
        return tau

    def run(self, parameter=None):
        uniques = np.unique(self.train)
        pack = list(itertools.product(range(self.train.shape[0]),range(self.test.shape[0])))
        pack = list(zip(*pack))
        scores = np.array(list(map(lambda x,y: self.distance(x,y, uniques), pack[0], pack[1])))
        return scores 
    
class CLUSA(Evaluation):
    
    theta_AUR_PR = 1
    theta_AUC_ROC = 2
    
    def __init__(self):
        self.b = 10
        self.theta = 1
        self.weights = None
        self.to_weigh = True
        
    def set_to_weigh(self, to_weigh):
        self.to_weigh = to_weigh
        return self
    
    def set_b(self, b):
        self.b = b
        return self
    
    def set_theta(self, theta):
        self.theta = theta
        return self
    
    def set_weights(self, weights):
        self.weights = weights
        return self
    
    def distance(self, idx_train, idx_test, parameter=None):
        def label_filter(dataset, minimum = 0):
            minimum = 2
            return np.array(list(filter(lambda x: np.count_nonzero(np.bincount(x)!=0)>=minimum, dataset))).astype(np.int16)

        def auc_pr_curve(a, b):
            x = precision_recall_curve(a, b)
            return auc(x[1],x[0])
        
        if self.theta == CLUSA.theta_AUR_PR:
            e = np.concatenate(np.array(list(map(lambda ix: label_filter(np.array(list(map(lambda i: (self.train[ix,:] > i).astype(int), set(self.train[ix,:])))), minimum = 2), idx_train))))
            scores = np.array(list(map(lambda g: auc_pr_curve(g, self.test[idx_test]), e)))
            weight = np.array(list(map(lambda g: np.bincount(g)[0]/float(len(g)), e)))
            return np.array((weight, scores))
        elif self.theta == CLUSA.theta_AUR_ROC:
            e = np.concatenate(np.array(list(map(lambda ix: label_filter(np.array(list(map(lambda i: (self.train[ix,:] > i).astype(int), set(self.train[ix,:])))), minimum = 2), idx_train))))
            scores = np.array(list(map(lambda g: roc_auc_score(g, self.test[idx_test]), e)))
            if self.weights == None:             
                weight = np.array(list(map(lambda g: np.bincount(g)[0]/float(len(g)), e)))
            else:
                weight = self.weights
            return np.array((weight, scores))
        else:
            raise Exception("Set a valid theta function")
        

    def run(self, parameter=None, std=False):
        with warnings.catch_warnings():
            uniques = np.unique(self.train)
            warnings.simplefilter("ignore", category=RuntimeWarning)
            scores = self.distance(range(self.train.shape[0]), 0, uniques)
            weights = scores[0,:]
            values = scores[1,:]
            bins = np.array(range(0,self.b+1))/float(self.b)
            idx = np.digitize(weights, bins)
            bin_scores = np.array([values[idx==x].mean() for x in range(1,len(bins))])
            if self.to_weigh == False:
                return bin_scores
            bin_compress = np.array(range(0,self.b))/float(self.b)+0.05
            scores_a = np.nansum((bin_scores)*(bin_compress/(bin_compress.sum())))
            return scores_a
      
    

