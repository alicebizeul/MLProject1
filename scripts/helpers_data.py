#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 11:59:33 2019
@author: Juliane
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

from implementations import *
from helpers_optimization import *
from helpers_visualization import *

# =============================================================================
# Load and submit data
# =============================================================================

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample, takes every 50 sample from the data set only if argument sub_sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

# =============================================================================
# Creation of feature matrix 
# =============================================================================

def split_subsets(tX, y, labels_feat):
    # Splitting the dataset based on the value of PRI_jet_num
    PRI_jet_num = tX[:, 22]
    
    mask_ss0 = PRI_jet_num == 0
    ss0_tX = tX[mask_ss0]
    ss0_y = y[mask_ss0]
    print("Subset 0 contains {} samples ".format(mask_ss0.sum()))
    
    mask_ss1 = PRI_jet_num == 1
    ss1_tX = tX[mask_ss1]
    ss1_y = y[mask_ss1]
    print("Subset 1 contains {} samples ".format(mask_ss1.sum()))
    
    mask_ss2 = PRI_jet_num == 2
    ss2_tX = tX[mask_ss2]
    ss2_y = y[mask_ss2]
    print("Subset 2 contains {} samples ".format(mask_ss2.sum()))
    
    mask_ss3 = PRI_jet_num == 3
    ss3_tX = tX[mask_ss3]
    ss3_y = y[mask_ss3]
    print("Subset 3 contains {} samples ".format(mask_ss3.sum()))
    
    ss0_tX, ss1_tX, ss2_tX, ss3_tX, labels_feat = remove_undef_feat(ss0_tX, ss1_tX, ss2_tX, ss3_tX,labels_feat)

    return ss0_tX, ss0_y, ss1_tX, ss1_y, ss2_tX, ss2_y, ss3_tX, ss3_y, labels_feat


def split_subsets_test(tX,labels_feat):
    # Splitting the dataset based on the value of PRI_jet_num
    PRI_jet_num = tX[:, 22]
    
    mask_ss0 = PRI_jet_num == 0
    ss0_tX = tX[mask_ss0]
    print("Subset 0 contains {} samples ".format(mask_ss0.sum()))
    
    mask_ss1 = PRI_jet_num == 1
    ss1_tX = tX[mask_ss1]
    print("Subset 1 contains {} samples ".format(mask_ss1.sum()))
    
    mask_ss2 = PRI_jet_num == 2
    ss2_tX = tX[mask_ss2]
    print("Subset 2 contains {} samples ".format(mask_ss2.sum()))
    
    mask_ss3 = PRI_jet_num == 3
    ss3_tX = tX[mask_ss3]
    print("Subset 3 contains {} samples ".format(mask_ss3.sum()))
    
    ss0_tX, ss1_tX, ss2_tX, ss3_tX, labels_feat = remove_undef_feat(ss0_tX, ss1_tX, ss2_tX, ss3_tX, labels_feat)

    return ss0_tX, mask_ss0, ss1_tX, mask_ss1, ss2_tX, mask_ss2, ss3_tX, mask_ss3, labels_feat


def remove_undef_feat(ss0_tX, ss1_tX, ss2_tX, ss3_tX, labels_feat):
    
    #Now, we can remove the categorical feature "PRI_jet_num" for our subsets
    ss0_tX = np.delete(ss0_tX, 22, axis=1)
    ss1_tX = np.delete(ss1_tX, 22, axis=1)
    ss2_tX = np.delete(ss2_tX, 22, axis=1)
    ss3_tX = np.delete(ss3_tX, 22, axis=1)
    
    labels_feat2 = np.delete(labels_feat,22)
    print('Remaining features for subset 2, 3: {}'.format(labels_feat2))
    
    # Removing undefined features for the corresponding subsets
    features_undefined_ss01 = [4, 5, 6, 12, 25, 26, 27]
    ss0_tX = np.delete(ss0_tX, features_undefined_ss01, axis=1)
    ss1_tX = np.delete(ss1_tX, features_undefined_ss01, axis=1)
    
    labels_feat1 = np.delete(labels_feat2,[4,5,6,12,25,26,27])
    print('Remaining features for subset 1: {}'.format(labels_feat1))
         
    features_undefined_ss0 = [18, 19, 20, 21] # taking into account indices of the features previously removed
    ss0_tX = np.delete(ss0_tX, features_undefined_ss0, axis=1)
    
    labels_feat0 = np.delete(labels_feat1,[18,19,20, 21])
    print('Remaining features for subset 0: {}'.format(np.delete(labels_feat1,[18,19,20, 21])))
    
    labels_feat = [labels_feat0, labels_feat1, labels_feat2, labels_feat2]
    
    return ss0_tX, ss1_tX, ss2_tX, ss3_tX, labels_feat


def remove_correlated_feat(ss0_tX, ss1_tX, ss2_tX, ss3_tX, labels_feat):
    
    # Subset 0 - keep DER_pt_tot
    print("Deleted features for subset 0 : {}".format(labels_feat[0][3]))
    ss0_tX = np.delete(ss0_tX, 3, axis=1)
    labels_feat[0] = np.delete(labels_feat[0],3)
    
    # Subset 1 - keep DER_sum_pt
    correlated_features_ss1 = [3, 18, 21]
    print("Deleted features for subset 1 : {}".format(labels_feat[1][correlated_features_ss1]))
    ss1_tX = np.delete(ss1_tX, correlated_features_ss1, axis=1)
    labels_feat[1] = np.delete(labels_feat[1],correlated_features_ss1)
    
    # Subset 2 - keep DER_sum_pt
    correlated_features_ss2 = [21, 22, 28]
    print("Deleted features for subset 2 : {}".format(labels_feat[2][correlated_features_ss2]))
    ss2_tX = np.delete(ss2_tX, correlated_features_ss2, axis=1)
    labels_feat[2] = np.delete(labels_feat[2],correlated_features_ss2)
    
     # Subset 3 - keep PRI_jet_all_pt
    correlated_features_ss3 = [9, 21, 22]
    print("Deleted features for subset 3 : {}".format(labels_feat[3][correlated_features_ss3]))
    ss3_tX = np.delete(ss3_tX, correlated_features_ss3, axis=1)
    labels_feat[3] = np.delete(labels_feat[3],correlated_features_ss3)
    
    return ss0_tX, ss1_tX, ss2_tX, ss3_tX, labels_feat


def replace_undef_feat(tX,y,method, ref_median = []):
    
    if ref_median != []:
        undefined_indices = np.argwhere(tX == -999.0)
        tX_temp = np.delete(tX, undefined_indices[:,0], 0)
        y_change = np.copy(y)
        methode_computed = [] 
        if method == 'median' : 
            tX_change = np.copy(tX)
            tX_change[undefined_indices[:,0],0] = ref_median
            methode_computed = np.median(tX_temp[:,0])
        elif method == 'mean' : 
            tX_change = np.copy(tX)
            tX_change[undefined_indices[:,0],0] = ref_median
            methode_computed = np.mean(tX_temp[:,0])
        elif method == 'delete' : 
            tX_change = tX_temp 
            y_change = np.delete(y_change, undefined_indices[:,0],0) 
        
    elif ref_median == []:
        undefined_indices = np.argwhere(tX == -999.0)
        tX_temp = np.delete(tX, undefined_indices[:,0], 0)
        y_change = np.copy(y)
        methode_computed = [] 
        if method == 'median' : 
            tX_change = np.copy(tX)
            tX_change[undefined_indices[:,0],0] = np.median(tX_temp[:,0])
            methode_computed = np.median(tX_temp[:,0])
        elif method == 'mean' : 
            tX_change = np.copy(tX)
            tX_change[undefined_indices[:,0],0] = np.mean(tX_temp[:,0])
            methode_computed = np.mean(tX_temp[:,0])
        elif method == 'delete' : 
            tX_change = tX_temp 
            y_change = np.delete(y_change, undefined_indices[:,0],0)    
    
    return tX_change, y_change, methode_computed


def outliers_suppresion(subsample,y, std_number):

    deviation_feature = np.std(subsample,axis = 0)
    mean_feature = np.mean(subsample,axis = 0)
    index = []

    for i in range(np.size(subsample,1)):
        dev_idx = deviation_feature[i]
        mean_idx = mean_feature[i]
        threshold = (std_number*dev_idx) + mean_idx
        for j in range(np.size(subsample,0)):
            if abs(subsample[j,i]) > threshold:
                index.append(j)

    subsample_outliers = np.delete(subsample, index, 0)
    y_outliers = np.delete(y, index, 0)
    print("size of the dataset with {in_} and without {out} the outliers".format(in_=subsample.shape, out=subsample_outliers.shape))
    print("Number of sample suppressed oustide {std} std: {supp}".format(std=std_number, supp=(subsample.shape[0] - subsample_outliers.shape[0])))
    return subsample_outliers, y_outliers


def feature_processing(ss_tX, ss_y,replace_method, replace_feature = True, suppr_outliers = True, threshold = 3, ref_median=[]):
    
    methode_computed = []
    
    if replace_feature == True:  ss_tX, ss_y, methode_computed = replace_undef_feat(ss_tX, ss_y, method = replace_method, ref_median = [])
    if suppr_outliers == True: ss_tX, ss_y = outliers_suppresion(ss_tX, ss_y, threshold)
    
    return ss_tX, ss_y, methode_computed


def feat_augmentation(tx, threshold=0.003, train_set=True, index=[]):
    
    if train_set:
        corr_matrix = np.corrcoef(tx.T)
        index = np.argwhere(corr_matrix < threshold) #set threshold as an argument
    
    final_index = []
    for i in range(index.shape[0]):
        if index[i, 0] != index[i, 1]:
            final_index.append(index[i,:])
            
    if len(final_index)>0: 
        final_index = np.sort(final_index, axis=1) 
        final_index = np.unique(final_index, axis=0)

        for i in range(final_index.shape[0]):
            feat1  = tx[:,final_index[i][0]]
            feat2  = tx[:,final_index[i][1]]
            tx = np.c_[tx, np.multiply(feat1,feat2)]
    
    return tx, index 


def build_poly(tx, degree, feature_augmentation=False, tx_aug=[]):
    """Creation of feature matrix with vector of ones + features vector using the appropriate degree given by the argument"""
    tx_new = np.ones(tx.shape[0])
    if feature_augmentation:
        tx_new=np.c_[tx_new, tx_aug]
    for i in range(1,degree+1):
        tx_new = np.c_[tx_new,tx**i]
    return tx_new


def standardize(tx , mean = [], std = []):
    if mean == [] and std == []:
        mean = np.mean(tx[:,1:], axis = 0)
        std = np.std(tx[:,1:],axis = 0)
    return np.c_[tx[:,0],(tx[:,1:]- mean)/std], mean, std


def batch_iter(y, tx, batch_size, num_batches=1):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    """
    data_size = len(y)

    shuffle_indices = np.random.permutation(np.arange(data_size))
    shuffled_y = y[shuffle_indices]
    shuffled_tx = tx[shuffle_indices]
    
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
   

# =============================================================================
# Data exploration methods
# =============================================================================         
            
def compute_correlations(tX, y, labels, threshold=0.85, print_correlated_pairs=False, plot=False, save_fig=False):
    """Computes and plots a heatmap of the correlation matrix. This matrix comprises the Pearson correlation
    coefficients between (continuous) features and the Point-biserial coefficients between each feature and 
    the (categorical) output."""
    
    correlation_output = [cal_point_biserial_correlation(tX[:,i], y) for i in range(tX.shape[1])]
    correlation_features = np.corrcoef(tX.T) 
    corr_matrix = np.c_[correlation_features, correlation_output]
    
    # Plot
    if plot:
        figure = plt.figure(figsize=(20,20))
        ax = figure.add_subplot(111)
        cax = ax.matshow(corr_matrix, cmap=plt.cm.PuOr)
        figure.colorbar(cax)
        plt.xticks(range(len(labels)), labels, rotation=90)
        plt.yticks(range(len(labels)-1), labels[:-1])
        plt.tight_layout()
        plt.show()
        if save_fig:
            figure.savefig("CorrelationMatrix.png", bbox_inches='tight')
    
    # Rank feature importance based on correlation with output
    correlation_output = np.abs(correlation_output)
    ranked_index = correlation_output.argsort()
    ranked_features = [labels[i] for i in ranked_index]
    print("Ranked absolute correlation with output: ", np.sort(correlation_output))
    print("Ranked features: ", ranked_features)

    if print_correlated_pairs:
        #Print pairs of features highly correlated (above threshold)
        index = np.argwhere(correlation_features > threshold)
        final_index = []
        for i in range(index.shape[0]):
            if index[i, 0] != index[i, 1]:
                final_index.append(index[i,:])
         
        final_index = np.sort(final_index, axis=1) 
        final_index = np.unique(final_index, axis=0)
        
        print("\n Highly correlated features (correlation above {}) : {} ".format(threshold,labels[final_index]))
        print("Index: ", final_index)
        
    return ranked_index


def cal_point_biserial_correlation(x, y):
    """ Computes the point-biserial correlation coefficient between a continuous variable x
    and a dichotomous variable y.
    Here, y takes values in {-1, 1}."""
    
    group_b = x[y == -1]
    n_b = len(group_b)
    group_s = x[y == 1]
    n_s = len(group_s)
    
    n_x = len(x)
    
    mean_b = np.mean(group_b)
    mean_s = np.mean(group_s)
    
    coef = ((mean_s-mean_b)/np.std(x))*np.sqrt(n_b*n_s/(n_x*n_x))
    
    return coef