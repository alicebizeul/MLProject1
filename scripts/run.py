# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:17:40 2019

@author: Florent
"""
import numpy as np
from proj1_helpers import *
from implementations import * 
import os


#%% Load the training data into feature matrix, class labels, and event ids

DATA_TRAIN_PATH = os.path.dirname(os.getcwd()) + '/data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
labels_feature = np.genfromtxt(DATA_TRAIN_PATH, delimiter=",", dtype=str, max_rows=1)[2:]


#%% Feature Engineering

# Subsetting the dataset
ss0_tX, ss0_y, ss1_tX, ss1_y, ss2_tX, ss2_y, ss3_tX, ss3_y, labels_feat = split_subsets(tX, y,labels_feature)

#Dealing with undefined feature
ss0_tX = replace_undef_feat(ss0_tX,method = 'median')
ss1_tX = replace_undef_feat(ss1_tX,method = 'median')
ss2_tX = replace_undef_feat(ss2_tX,method = 'median')
ss3_tX = replace_undef_feat(ss3_tX,method = 'median')

#Dealing with outiliers
_=outliers_suppresion(ss0_tX, 3)
_=outliers_suppresion(ss1_tX, 3)
_=outliers_suppresion(ss2_tX, 3)
_=outliers_suppresion(ss3_tX, 3)


#%%Feature selection

labels_0, labels_1, labels_2, labels_3 = labels_feat
# SUBSET 0
ranked_index_ss0, ranked_features_ss0 = plot_correlation_matrix(ss0_tX, ss0_y, np.append(labels_0,'Output'), "CorrelationMatrix_ss0.png")
# SUBSET 1
ranked_index_ss1, ranked_features_ss1 = plot_correlation_matrix(ss1_tX, ss1_y, np.append(labels_1,'Output'), "CorrelationMatrix_ss1.png")
# SUBSET 2
ranked_index_ss2, ranked_features_ss2 = plot_correlation_matrix(ss2_tX, ss2_y, np.append(labels_2,'Output'), "CorrelationMatrix_ss2.png")
# SUBSET 3
ranked_index_ss3, ranked_features_ss3 = plot_correlation_matrix(ss3_tX, ss3_y, np.append(labels_3,'Output'), "CorrelationMatrix_ss3.png")


#%% Model selction and weights computation


loss_tr, loss_te, w0 = cross_validation_demo(ss0_y, ss0_tX, degree = 1, seed = 5, k_fold = 5, class_distribution = False, error = 'rmse', method = 'ls')
loss_tr, loss_te, w1 = cross_validation_demo(ss1_y, ss1_tX, degree = 1, seed = 5, k_fold = 5, class_distribution = False, error = 'rmse', method = 'ls')
loss_tr, loss_te, w2 = cross_validation_demo(ss2_y, ss2_tX, degree = 1, seed = 5, k_fold = 5, class_distribution = False, error = 'rmse', method = 'ls')
loss_tr, loss_te, w3 = cross_validation_demo(ss3_y, ss3_tX, degree = 1, seed = 5, k_fold = 5, class_distribution = False, error = 'rmse', method = 'ls')



#%% Generate predictions and save ouput in csv format for submission

weights0 = np.ones(ss0_tX_test.shape[1])
weights1 = np.ones(ss1_tX_test.shape[1])
weights2 = np.ones(ss2_tX_test.shape[1])
weights3 = np.ones(ss3_tX_test.shape[1])

# Subset 0 
y_pred0 = predict_labels(weights0, ss0_tX_test)
#Subset 1
y_pred1 = predict_labels(weights1, ss1_tX_test)
#Subset 2
y_pred2 = predict_labels(weights2, ss2_tX_test)
#Subset 3
y_pred3 = predict_labels(weights3, ss3_tX_test)

#Stack all prediction from 4 subgroups to get y_pred in corredt order 
y_pred = np.ones(len(ids_test))
y_pred[index0] = y_pred0
y_pred[index1] = y_pred1
y_pred[index2] = y_pred2
y_pred[index3] = y_pred3

OUTPUT_PATH = os.path.dirname(os.getcwd()) + '/data/' + str(datetime.now())
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

#%%







#%%

