# -*- coding: utf-8 -*-
"""
Created on Wed May 17 14:09:50 2017

@author: rebli
"""

import itertools
import numpy as np
import os
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from multiprocessing import Pool
from timeit import default_timer as timer
from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd

def log():
    logFormatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt='%m/%d %I:%M:%S')
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)
    
    if not rootLogger.handlers:
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        rootLogger.addHandler(consoleHandler)
    return rootLogger

rootLogger  = log()    

def generate_candidates(data, max_len=5, min_len=2):
    candidates, l = [], max_len
    while l >= min_len:
        for i in np.arange(0, len(data)):
            _data, _label = data[i][0], data[i][1]
            for k in np.arange(0, len(_data)-l+1): 
                candidates.append((_data[k:k+l], _label))
        l -= 1
    return candidates

def worker(_data, _label, shapelet):

    d, idx = subsequence_dist(_data, shapelet)
    return d, _data, _label


def check_candidate(data, shapelet):
    histogram = {}
     
    pool = Pool(processes=4)
    def aggregator(res): 
            if res[0] not in histogram:
                histogram[res[0]] = []
            histogram[res[0]].append((res[1], res[2]))

    [pool.apply_async(worker, args=(entry[0],entry[1],shapelet), callback=aggregator)  for entry in data]          
    pool.close()
    pool.join()

    return find_best_split_point(histogram)

def calculate_dict_entropy(data):
    counts = {}
    for entry in data:
        if entry[1] in counts: counts[entry[1]] += 1
        else: counts[entry[1]] = 1
    return calculate_entropy(np.divide(list(counts.values()), float(sum(list(counts.values())))))

def worker_1(distance, histogram):

    data_left = []
    data_right = []
    for distance2 in histogram:
        if distance2 <= distance: 
            data_left.extend(histogram[distance2])
        else: 
            data_right.extend(histogram[distance2])
    return data_left, data_right, distance
   

def calculate_bins(histogram):
    
    trans = KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='uniform')
    var_array = np.array(list(histogram.keys())).reshape(-1,1) 
    val = trans.fit_transform(var_array)
    histogram_new = dict()

    for _a, _b in zip(list(val.reshape(1,-1)[0]), list(var_array.reshape(1,-1)[0])):
        if _a not in histogram_new:
           histogram_new[_a] = []

        histogram_new[_a].append(histogram[_b][0]) 

    return histogram_new
    
def find_best_split_point(histogram):

    histogram_values = list(itertools.chain.from_iterable(list(histogram.values())))
    prior_entropy = calculate_dict_entropy(histogram_values)
    best_distance, max_ig = 0, 0
    best_left, best_right = None, None

    histogram_bins = calculate_bins(histogram)
    
    for distance in histogram_bins:
        data_left = []
        data_right = []
        for distance2 in histogram_bins:
            if distance2 <= distance: 
                data_left.extend(histogram_bins[distance2])
            else: 
                data_right.extend(histogram_bins[distance2])
 
        ig = prior_entropy - (float(len(data_left))/float(len(histogram_values))*calculate_dict_entropy(data_left) + \
             float(len(data_right))/float(len(histogram_values)) * calculate_dict_entropy(data_right))
        if ig > max_ig: 
            best_distance, max_ig, best_left, best_right = distance, ig, data_left, data_right
    return max_ig, best_distance, best_left, best_right

def manhattan_distance(a, b, min_dist=float('inf')):
    dist = 0
    for x, y in zip(a, b):
        dist += np.abs(float(x)-float(y))
        if dist >= min_dist: return None
    return dist

def calculate_entropy(probabilities):
    return sum([-prob * np.log(prob)/np.log(2) if prob != 0 else 0 for prob in probabilities])

def subsequence_dist(time_serie, sub_serie):
    if len(sub_serie) < len(time_serie):
        min_dist, min_idx = float("inf"), 0

        for i in np.arange(0, len(time_serie)-len(sub_serie)+1):
            dist = manhattan_distance(sub_serie, time_serie[i:i+len(sub_serie)], min_dist)
            if dist is not None and dist < min_dist: 
                min_dist, min_idx = dist, i
        return min_dist, min_idx
    else:
        return None, None

def find_shapelets_bf(data, max_len=100, min_len=1, plot=True, verbose=True):

    rootLogger.info(f'Generating the condidates...')
    candidates = generate_candidates(data, max_len, min_len)

    rootLogger.info(f'Total Candidates: {len(candidates)}')
    bsf_gain, bsf_shapelet = 0, None
    bsf_dist = 0
    if verbose: 
        candidates_length = len(candidates)

    for idx, candidate in enumerate(candidates):
        gain, dist, data_left, data_right = check_candidate(data, candidate[0])
        if verbose: 
            rootLogger.info(f'{idx} / {candidates_length}: Gain: {gain} Dist: {dist}')
        if gain > bsf_gain:
            bsf_gain, bsf_shapelet, bsf_dist = gain, candidate[0], dist
            
            if verbose:
                rootLogger.info(f'Found new best shapelet with gain & dist: {bsf_gain}/{dist}')
                
    return bsf_shapelet, bsf_dist












