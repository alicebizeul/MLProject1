#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 11:54:42 2019
@author: Juliane
"""

import numpy as np

# =============================================================================
# Loss functions
# =============================================================================

def compute_loss(y, tx, w, error, lambdas=[]):
    """Calculate the loss.
    You can calculate the loss using method given in arguments.
    """
    y_pred = predict_labels(w,tx)
    
    if error == 'mse': return cal_mse(cal_error(y,y_pred))
    elif error == 'rmse': return cal_rmse(cal_error(y,y_pred))
    elif error == 'class': return cal_classerror(y,y_pred)
    elif error == 'classification':return cal_classificationerror(y,y_pred)
    elif error == 'logl':return cal_loglike(y, tx, w)
    elif error == 'logl_r': return cal_loglike_r(y,tx,w,lambdas)
    else: raise NotImplementedError

def cal_error(y, y_pred):
    """ Returns vector of 0,2 or -2, the difference between vector of labels and vector of predicted labels"""
    return y - y_pred
    

def cal_mse(error):
    """Returns the mean square error for vector e."""
    return 1/2*np.mean(error**2)


def cal_rmse(error):
    """Returns the root mean square error using the mean square error as input """
    return np.sqrt(2*cal_mse(error))


def cal_classerror(y,y_pred):
    """Returns the class error (percentage of fails) which takes 
    reequilibrates class distribution into acount"""
    class1 = np.sum(y_pred[y ==1] != 1)/np.sum(y == 1)
    class2 = np.sum(y_pred[y == -1] != -1)/np.sum(y == -1)
    return 0.5*(class1 + class2)


def cal_classificationerror(y, y_pred):
    """Returns the classification error = percentage of fails, does not 
    take class distribution among data set into account """
    return 1-accuracy(y,y_pred)
    

def cal_loglike(y, tx, w):
    """Compute the cost by negative log likelihood"""
    delta = 0
    sigmoid_fct = sigmoid(tx.dot(w))
    if len(sigmoid_fct[sigmoid_fct==0])> 0 or len(sigmoid_fct[(1-sigmoid_fct)==0]) > 0:
        delta = 0.0000000001
    loss = y.T.dot(np.log(sigmoid_fct+delta)) + (1 - y).T.dot(np.log(1 - sigmoid_fct+delta))
    return np.squeeze(-loss)


def cal_loglike_r(y, tx, w, lambda_):
     return cal_loglike(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
 
    
def sigmoid(z):
    return np.exp(z) / (1 + np.exp(z))
 
# =============================================================================
# Gradients
# =============================================================================    
 
def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def compute_logistic_gradient(y, tx, w):
    """compute the gradient of loss."""
    sigmo = sigmoid(tx.dot(w))
    return tx.T.dot(sigmo - y)
 
# =============================================================================
# Cross validation
# =============================================================================  


def build_k_indices(y, k_fold, seed):
    """Create the k-folds"""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation_demo(y, x, degree, seed, k_fold = 4, class_distribution = False, error ='class', method='rr', feature_augmentation=False, hyperparams=[]):
    
    if class_distribution == True : y, x = equal_class(y,x)
    k_indices = build_k_indices(y, k_fold, seed)
           
    verify_proportion(y,k_indices)
    
    # cross validation
    loss_tr, loss_te, w, accuracy = choose_method(y, x, degree, seed, k_fold, k_indices, error, method, feature_augmentation, hyperparams)
    
    return loss_tr, loss_te, w, accuracy


def choose_method(y, x, degree, seed, k_fold = 4, k_indices = [], error ='class', method='rr', feature_augmentation=False, hyperparams=[]):
    
    loss_tr = []
    loss_te = []
    
    w = []
    accuracy = []
    
    if method == 'rr':
        for lambda_ in hyperparams[0]:
            loss_tr_tmp, loss_te_tmp, w_tmp, acc_tmp  = single_cross_val(y, x, degree, k_fold, k_indices, method,error,feature_augmentation, [lambda_])
            loss_tr.append(loss_tr_tmp)
            loss_te.append(loss_te_tmp)
            accuracy.append(acc_tmp)
            w.append(w_tmp)
    elif method == 'ls':
        loss_tr_tmp, loss_te_tmp, w_tmp, acc_tmp= single_cross_val(y, x, degree, k_fold, k_indices, method,error, feature_augmentation)
        loss_tr.append(loss_tr_tmp)
        loss_te.append(loss_te_tmp)
        accuracy.append(acc_tmp)
        w.append(w_tmp)
    elif method =='lsGD':
        for gamma in hyperparams[2]:
            loss_tr_tmp, loss_te_tmp, w_tmp, acc_tmp = single_cross_val(y, x, degree, k_fold, k_indices, method,error, feature_augmentation, [hyperparams[0],hyperparams[1],gamma])
            loss_tr.append(loss_tr_tmp)
            loss_te.append(loss_te_tmp)
            accuracy.append(acc_tmp)
            w.append(w_tmp)
    elif method =='lsSGD':
        for gamma in hyperparams[2]:
            loss_tr_tmp, loss_te_tmp, w_tmp, acc_tmp = single_cross_val(y, x, degree, k_fold, k_indices, method,error,feature_augmentation,[hyperparams[0],hyperparams[1],gamma,hyperparams[3]])
            loss_tr.append(loss_tr_tmp)
            loss_te.append(loss_te_tmp)
            accuracy.append(acc_tmp)
            w.append(w_tmp)
    elif method == 'log':
        for gamma in hyperparams[2]:
            loss_tr_tmp, loss_te_tmp, w_tmp, acc_tmp = single_cross_val(y, x, degree, k_fold, k_indices, method,error,feature_augmentation, [hyperparams[0],hyperparams[1],gamma])
            loss_tr.append(loss_tr_tmp)
            loss_te.append(loss_te_tmp)
            accuracy.append(acc_tmp)
            w.append(w_tmp)
    elif method == 'rlog':
        for lambda_ in hyperparams[3]:
            for gamma in hyperparams[2]:
                loss_tr_tmp, loss_te_tmp, w_tmp, acc_tmp = single_cross_val(y, x, degree, k_fold, k_indices, method,error,feature_augmentation, [hyperparams[0],hyperparams[1],gamma,lambda_])
            loss_tr.append(loss_tr_tmp)
            loss_te.append(loss_te_tmp)
            accuracy.append(acc_tmp)
            w.append(w_tmp)
    else: raise NotImplementedError
    return loss_tr, loss_te, w, accuracy


def single_cross_val(y, x, degree, k_fold, k_indices, method, error, feature_augmentation=False, hyperparams = []):
    loss_tr_tmp = []
    loss_te_tmp = []
    w_tmp = []
    accuracy = []
    
    for k in range(k_fold):
        loss_tr, loss_te, w , acc = cross_validation(y, x, degree, k, k_indices, method, error, feature_augmentation, hyperparams)
        loss_tr_tmp.append(loss_tr)
        loss_te_tmp.append(loss_te)    
        w_tmp.append(w)
        accuracy.append(acc)
    
    if not feature_augmentation:
        w_mean = np.mean(w_tmp,axis=0)
    else : 
        w_mean=[]
    #print("Accuracy = {}".format(accuracy))
    return loss_tr_tmp, loss_te_tmp, w_mean, accuracy


def cross_validation(y, x, degree, k, k_indices,method, error, feature_augmentation, hyperparams):
    """return the loss of ridge regression."""
    from helpers_data import feature_processing, feat_augmentation, standardize, build_poly
    from implementations import ridge_regression, least_squares, least_squares_GD, least_squares_SGD, logistic_regression, reg_logistic_regression
    
    
    # get k'th subgroup in test, others in train
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    
    y_te = y[te_indice]
    y_tr = y[tr_indice]
    x_te = x[te_indice]
    x_tr = x[tr_indice]
    
    x_tr, y_tr, median = feature_processing (x_tr, y_tr, 'mean', replace_feature = True, suppr_outliers = hyperparams[-1], threshold = 3, ref_median=[])
    x_te, y_te, _= feature_processing (x_te, y_te, 'mean', replace_feature = True, suppr_outliers = False, threshold = 3, ref_median=median)
    
    
    tx_tr_aug = []
    tx_te_aug = []
    if feature_augmentation:
        tx_tr_aug, index = feat_augmentation(x_tr, 0.003)
        tx_te_aug, _ = feat_augmentation(x_te, 0.003, False, index)
    
    # form data with polynomial degree
    tx_tr = build_poly(x_tr, degree, feature_augmentation, tx_tr_aug)
    tx_te = build_poly(x_te, degree, feature_augmentation, tx_te_aug)
    tx_tr, mean, std = standardize(tx_tr)
    tx_te, _, _ = standardize(tx_te, mean, std)
    
    #print('Mean and std of each feature in train set: {} , {}'.format(tx_tr.mean(axis = 0),tx_tr.std(axis = 0)))
    #print('Mean and std of each feature in test set: {} , {}'.format(tx_te.mean(axis = 0),tx_te.std(axis = 0)))
    
    
    
    if method == 'rr': w,_ = ridge_regression(y_tr, tx_tr, hyperparams[0]) # ridge regression
    elif method == 'ls': w,_ = least_squares(y_tr, tx_tr) # least square
    elif method == 'lsGD': w,_ = least_squares_GD(y_tr, tx_tr, hyperparams[0], hyperparams[1], hyperparams[2]) # gradient descent
    elif method == 'lsSGD': w,_ = least_squares_SGD(y_tr, tx_tr, hyperparams[0], hyperparams[1], hyperparams[2], hyperparams[3]) # stoch GD
    elif method == 'log': w,_ = logistic_regression(y_tr, tx_tr, hyperparams[0], hyperparams[1], hyperparams[2]) # logistic reg
    elif method == 'rlog': w,_ =reg_logistic_regression(y_tr, tx_tr, hyperparams[3], np.zeros(tx_tr.shape[1]), hyperparams[1], hyperparams[2]) # regularised logistic reg
    else: raise NotImplementedError
   
    if method == 'log':
        loss_tr = cal_loglike(y_tr, tx_tr, w)
        loss_te = cal_loglike(y_te, tx_te, w)
    elif method == 'rlog':
        loss_tr = cal_loglike_r(y_tr, tx_tr, w, hyperparams[3])
        loss_te = cal_loglike_r(y_te, tx_te, w, hyperparams[3])
    else :
        # calculate the loss for train and test data
        loss_tr = compute_loss(y_tr, tx_tr, w, error)
        loss_te = compute_loss(y_te, tx_te, w, error)      
    
    y_pred = predict_labels(np.array(w).T, tx_te)
    acc = accuracy(y_te,y_pred)
    
    return loss_tr, loss_te, w, acc


def cross_validation_demo_featselect(y, x, labels, degree, seed, k_fold = 4, class_distribution = False, error ='class', method='rr', feature_augmentation=False, hyperparams=[]):
    
    from helpers_data import compute_correlations
    
    ranked_index=compute_correlations(x, y, labels, plot=False)
    x = np.fliplr(x[:,ranked_index])
    
    if class_distribution == True : y, x = equal_class(y,x)
    k_indices = build_k_indices(y, k_fold, seed)
           
    verify_proportion(y,k_indices)
    
    loss_tr = []
    loss_te = []
    w = []
    accuracy = []
    
    # cross validation
    for feat in range(1,x.shape[1]+1):
        x_croped = x[:,:feat]
        print('Number of best features tested : {}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'.format(x_croped.shape[1]))
        loss_tr_tmp, loss_te_tmp, w_tmp, accuracy_tmp = choose_method(y, x_croped, degree, seed, k_fold, k_indices, error, method, feature_augmentation, hyperparams)
        loss_tr.append(loss_tr_tmp)
        loss_te.append(loss_te_tmp)
        w.append(w_tmp)
        accuracy.append(accuracy_tmp)
            
    #cross_validation_visualization(hyperparams, loss_tr, loss_te) #A MODIFIER    
    return loss_tr, loss_te, w, accuracy


def equal_class(y,x):
    y_class0 = y[y==-1]
    y_class1 = y[y==1]
        
    x_class0 = x[y==-1][:]
    x_class1 = x[y==1][:]
        
    to_keep = np.random.permutation(len(y_class0))[:(len(y_class1)-1)]
    return  np.concatenate((y_class0[to_keep],y_class1),axis = 0), np.concatenate((x_class0[to_keep][:],x_class1),axis = 0)
      
    
def verify_proportion(y,k_indices):
    print('Number of remaining samples before start cross val : {}'.format(len(y)))
    print("Proportion of Bosons in all train set : {} %".format(100*len(y[y==1])/len(y)))
    print("Proportion of Bosons in test fold 1: {} %".format(100*len(y[k_indices[0]][y[k_indices[0]]==1])/len(y[k_indices[0]])))


# =============================================================================
# Performance assessment
# =============================================================================    
    
def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def accuracy(y,y_pred):
    """ Returns accuracy of classification = percentage of success"""
    return np.sum(y == y_pred)/len(y)
