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


#%% Feature Engineering

# Subsetting the dataset
ss0_tX, ss0_y, ss1_tX, ss1_y, ss2_tX, ss2_y, ss3_tX, ss3_y, labels_feat = split_subsets(tX, y,labels_feature)

#Dealing with undefined feature
ss0_tX, ss0_y = replace_undef_feat(ss0_tX,ss0_y,method = 'delete')
ss1_tX, ss1_y = replace_undef_feat(ss1_tX,ss1_y,method = 'delete')
ss2_tX, ss2_y = replace_undef_feat(ss2_tX,ss2_y,method = 'delete')
ss3_tX, ss3_y = replace_undef_feat(ss3_tX,ss3_y,method = 'delete')

print(ss3_tX.shape)

#Dealing with outiliers
ss0_tX, ss0_y = outliers_suppresion(ss0_tX, ss0_y, 3)
ss1_tX, ss1_y = outliers_suppresion(ss1_tX, ss1_y, 3)
ss2_tX, ss2_y = outliers_suppresion(ss2_tX, ss2_y, 3)
ss3_tX, ss3_y = outliers_suppresion(ss3_tX, ss3_y, 3)

print(ss3_tX.shape)

#%%Feature selection

labels_0, labels_1, labels_2, labels_3 = labels_feat
# SUBSET 0
ss0_tX3, ss1_tX3, ss2_tX3, ss3_tX3, labels_feat = remove_correlated_feat(ss0_tX3, ss1_tX3, ss2_tX3, ss3_tX3, labels_feat)


#%% Model selction and weights computation



#%% Final Training on full data set

final_degree = 1 # TO UPDATE 

# CHANGE THE ss0_tX variable depending on the features to use
ss0_tX_train = build_poly(ss0_tX, final_degree)
ss1_tX_train = build_poly(ss1_tX, final_degree)
ss2_tX_train = build_poly(ss2_tX, final_degree)
ss3_tX_train = build_poly(ss3_tX, final_degree)

# Standardisation
ss0_tX_train, mean0, std0 = standardize(ss0_tX_train)
ss1_tX_train, mean1, std1 = standardize(ss1_tX_train)
ss2_tX_train, mean2, std2 = standardize(ss2_tX_train)
ss3_tX_train, mean3, std3 = standardize(ss3_tX_train)

# TO UPDATE !!!!  
#find optimal weights for the entire train set
_,_, weights0 = cross_validation_demo(ss0_y, ss0_tX, degree = 1, seed = 5, k_fold = 5, class_distribution = False, error = 'rmse', method = 'ls')
_,_, weights1 = cross_validation_demo(ss1_y, ss1_tX, degree = 1, seed = 5, k_fold = 5, class_distribution = False, error = 'rmse', method = 'ls')
_,_, weights2 = cross_validation_demo(ss2_y, ss2_tX, degree = 1, seed = 5, k_fold = 5, class_distribution = False, error = 'rmse', method = 'ls')
_,_, weights3 = cross_validation_demo(ss3_y, ss3_tX, degree = 1, seed = 5, k_fold = 5, class_distribution = False, error = 'rmse', method = 'ls')


#%% Load the test set

DATA_TEST_PATH = os.path.dirname(os.getcwd()) + '/data/test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

#%% Generate predictions and save ouput in csv format for submission

#Splitting the test set
ss0_tX_test, index0, ss1_tX_test, index1, ss2_tX_test, index2, ss3_tX_test, index3 = split_subsets_test(tX_test, labels_feature)


#%%

#Build the model
ss0_tX_test = build_poly(ss0_tX_test, final_degree)
ss1_tX_test = build_poly(ss1_tX_test, final_degree)
ss2_tX_test = build_poly(ss2_tX_test, final_degree)
ss3_tX_test = build_poly(ss3_tX_test, final_degree)

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
OUTPUT_PATH = os.path.dirname(os.getcwd()) + '\\data\\Buzz2.csv'
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

