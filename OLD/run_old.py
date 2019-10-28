# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 19:35:33 2019
@author: Florent
"""
import numpy as np
import os
from implementations import * 
from helpers_optimization import *
from helpers_data import *
from helpers_visualization import *

#%% Load the training set into feature matrix, class labels, and event ids

DATA_TRAIN_PATH = os.path.dirname(os.getcwd()) + '/data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
labels_feature = np.genfromtxt(DATA_TRAIN_PATH, delimiter=",", dtype=str, max_rows=1)[2:]

#%% Load the test set

DATA_TEST_PATH = os.path.dirname(os.getcwd()) + '/data/test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

#%% Feature Engineering

# Subsetting the dataset
ss0_tX, ss0_y, ss1_tX, ss1_y, ss2_tX, ss2_y, ss3_tX, ss3_y, labels_feat = split_subsets(tX, y,labels_feature)

ss0_tX, ss0_y, median0 = feature_processing (ss0_tX, ss0_y, 'median', replace_feature = True, suppr_outliers = False, threshold = 3, ref_median=[])
ss1_tX, ss1_y, median1 = feature_processing (ss1_tX, ss1_y, 'median', replace_feature = True, suppr_outliers = False, threshold = 3, ref_median=[])
ss2_tX, ss2_y, median2 = feature_processing (ss2_tX, ss2_y, 'median', replace_feature = True, suppr_outliers = False, threshold = 3, ref_median=[])
ss3_tX, ss3_y, median3 = feature_processing (ss3_tX, ss3_y, 'median', replace_feature = True, suppr_outliers = False, threshold = 3, ref_median=[])
    

#%% Hyperparameters initiation

lambdas = [2.15443469e-15,7.19685673e-16,2.15443469e-15,1.0e-15]
final_degree = [16,15,16,14]


#%% Final Training on full data set


ss0_tX_train_aug, index_0 = feat_augmentation(ss0_tX, 0.003, True)
ss1_tX_train_aug, index_1 = feat_augmentation(ss1_tX, 0.003, True)
ss2_tX_train_aug, index_2 = feat_augmentation(ss2_tX, 0.003, True)
ss3_tX_train_aug, index_3 = feat_augmentation(ss3_tX, 0.003, True)

ss0_tX_train = build_poly(ss0_tX, final_degree[0], feature_augmentation=True, tx_aug=ss0_tX_train_aug)
ss1_tX_train = build_poly(ss1_tX, final_degree[1], feature_augmentation=True, tx_aug=ss1_tX_train_aug)
ss2_tX_train = build_poly(ss2_tX, final_degree[2], feature_augmentation=True, tx_aug=ss2_tX_train_aug)
ss3_tX_train = build_poly(ss3_tX, final_degree[3], feature_augmentation=True, tx_aug=ss3_tX_train_aug)

# Standardisation
ss0_tX_train, mean0, std0 = standardize(ss0_tX_train)
ss1_tX_train, mean1, std1 = standardize(ss1_tX_train)
ss2_tX_train, mean2, std2 = standardize(ss2_tX_train)
ss3_tX_train, mean3, std3 = standardize(ss3_tX_train)

# Model on the whole training set
weights0,_ = ridge_regression(ss0_y, ss0_tX_train, lambdas[0])
weights1,_ = ridge_regression(ss1_y, ss1_tX_train, lambdas[1])
weights2,_ = ridge_regression(ss2_y, ss2_tX_train, lambdas[2])
weights3,_ = ridge_regression(ss3_y, ss3_tX_train, lambdas[3])

#%% Splitting the set data

ss0_tX_test, index0, ss1_tX_test, index1, ss2_tX_test, index2, ss3_tX_test, index3, labels_feat = split_subsets_test(tX_test, labels_feature)

#%%

#Build the model
ss0_tX_test,_,_ = feature_processing (ss0_tX_test, ss0_y, 'median', replace_feature = True, suppr_outliers = False, threshold = 3, ref_median=median0)
ss1_tX_test,_,_ = feature_processing (ss1_tX_test, ss1_y, 'median', replace_feature = True, suppr_outliers = False, threshold = 3, ref_median=median1)
ss2_tX_test,_,_ = feature_processing (ss2_tX_test, ss2_y, 'median', replace_feature = True, suppr_outliers = False, threshold = 3, ref_median=median2)
ss3_tX_test,_,_ = feature_processing (ss3_tX_test, ss3_y, 'median', replace_feature = True, suppr_outliers = False, threshold = 3, ref_median=median3)

ss0_tX_test_aug, _ = feat_augmentation(ss0_tX_test, 0.003, False, index_0)
ss1_tX_test_aug, _ = feat_augmentation(ss1_tX_test, 0.003, False, index_1)
ss2_tX_test_aug, _ = feat_augmentation(ss2_tX_test, 0.003, False, index_2)
ss3_tX_test_aug, _ = feat_augmentation(ss3_tX_test, 0.003, False, index_3)

ss0_tX_test = build_poly(ss0_tX_test, final_degree[0], feature_augmentation=True, tx_aug=ss0_tX_test_aug)
ss1_tX_test = build_poly(ss1_tX_test, final_degree[1], feature_augmentation=True, tx_aug=ss1_tX_test_aug)
ss2_tX_test = build_poly(ss2_tX_test, final_degree[2], feature_augmentation=True, tx_aug=ss2_tX_test_aug)
ss3_tX_test = build_poly(ss3_tX_test, final_degree[3], feature_augmentation=True, tx_aug=ss3_tX_test_aug)

# standardize test data
ss0_tX_test, _, _ = standardize(ss0_tX_test, mean0, std0)
ss1_tX_test, _, _ = standardize(ss1_tX_test, mean1, std1)
ss2_tX_test, _, _ = standardize(ss2_tX_test, mean2, std2)
ss3_tX_test, _, _ = standardize(ss3_tX_test, mean3, std3)

# Subsets prediction
y_pred0 = predict_labels(np.array(weights0).T, ss0_tX_test)
y_pred1 = predict_labels(np.array(weights1).T, ss1_tX_test)
y_pred2 = predict_labels(np.array(weights2).T, ss2_tX_test)
y_pred3 = predict_labels(np.array(weights3).T, ss3_tX_test)

#Stack all prediction from 4 subgroups to get y_pred in corredt order 
y_pred = np.ones(len(ids_test))

y_pred[index0] = np.squeeze(y_pred0)
y_pred[index1] = np.squeeze(y_pred1)
y_pred[index2] = np.squeeze(y_pred2)
y_pred[index3] = np.squeeze(y_pred3)

#%%
OUTPUT_PATH = os.path.dirname(os.getcwd()) + '/data/BuzzLastyear.csv'
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
