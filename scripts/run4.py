# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:17:40 2019
@author: Florent
"""
import numpy as np
import os
from proj1_helpers import *
from implementations import * 
from datetime import datetime


#%% Load the training data into feature matrix, class labels, and event ids

DATA_TRAIN_PATH = os.path.dirname(os.getcwd()) + '/data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
labels_feature = np.genfromtxt(DATA_TRAIN_PATH, delimiter=",", dtype=str, max_rows=1)[2:]


#%% Load the data set

DATA_TEST_PATH = os.path.dirname(os.getcwd()) + '/data/test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

#%% Feature Engineering

# Subsetting the dataset
ss0_tX, ss0_y, ss1_tX, ss1_y, ss2_tX, ss2_y, ss3_tX, ss3_y, labels_feat = split_subsets(tX, y,labels_feature)

ss0_tX, ss0_y = feature_processing (ss0_tX, ss0_y, 'median', replace_feature = True, suppr_outliers = True, threshold = 3)
ss1_tX, ss1_y = feature_processing (ss1_tX, ss1_y, 'median', replace_feature = True, suppr_outliers = True, threshold = 3)
ss2_tX, ss2_y = feature_processing (ss2_tX, ss2_y, 'median', replace_feature = True, suppr_outliers = True, threshold = 3)
ss3_tX, ss3_y = feature_processing (ss3_tX, ss3_y, 'median', replace_feature = True, suppr_outliers = True, threshold = 3)
    

#%%Feature selection --> Juliane

#ss0_tX, ss1_tX, ss2_tX, ss3_tX,_ = remove_correlated_feat(ss0_tX, ss1_tX, ss2_tX, ss3_tX, labels_feat)
#ss0_tX_test, ss1_tX_test, ss2_tX_test, ss3_tX_test, labels_feat_test = remove_correlated_feat(ss0_tX_test, ss1_tX_test, ss2_tX_test, ss3_tX_test, labels_feat)

#%% Hyperparameters initiation

final_degree = [12,12,12,12] 
lambdas = [0.001,0.001,0.001,0.001]

#lambdas = [3.35981829e-10,1.0e-9,2.6e-8,1e-11]
#final_degree = [14,13,11,14] 


#%% Final Training on full data set
# CHANGE THE ss_tX variable depending on the features to use
ss0_tX_train = build_poly(ss0_tX, final_degree[0])
ss1_tX_train = build_poly(ss1_tX, final_degree[1])
ss2_tX_train = build_poly(ss2_tX, final_degree[2])
ss3_tX_train = build_poly(ss3_tX, final_degree[3])

# Standardisation
ss0_tX_train, mean0, std0 = standardize(ss0_tX_train)
ss1_tX_train, mean1, std1 = standardize(ss1_tX_train)
ss2_tX_train, mean2, std2 = standardize(ss2_tX_train)
ss3_tX_train, mean3, std3 = standardize(ss3_tX_train)

#Model on the whole set
weights0 = ridge_regression(ss0_y, ss0_tX_train, lambdas[0])
weights1 = ridge_regression(ss1_y, ss1_tX_train, lambdas[1])
weights2 = ridge_regression(ss2_y, ss2_tX_train, lambdas[2])
weights3 = ridge_regression(ss3_y, ss3_tX_train, lambdas[3])

#%% Generate predictions and save ouput in csv format for submission

#Splitting the test set
ss0_tX_test, index0, ss1_tX_test, index1, ss2_tX_test, index2, ss3_tX_test, index3, labels_feat = split_subsets_test(tX_test, labels_feature)


#%%

#Build the model
ss0_tX_test = build_poly(ss0_tX_test, final_degree[0])
ss1_tX_test = build_poly(ss1_tX_test, final_degree[1])
ss2_tX_test = build_poly(ss2_tX_test, final_degree[2])
ss3_tX_test = build_poly(ss3_tX_test, final_degree[3])

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
OUTPUT_PATH = os.path.dirname(os.getcwd()) + '\\data\\BuzzAller.csv'
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

