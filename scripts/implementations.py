#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 10:31:41 2019

@author: Juliane
"""

import numpy as np
from proj1_helpers import *
import matplotlib.pyplot as plt


# Creation of feature matrix 

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


def build_model_data(features, label):
    """Form (y,tX) to get regression data in matrix form."""
    y = label
    x = features
    num_samples = x.shape[0]
    tx = np.column_stack((x, np.ones(num_samples, dtype=x.dtype)))
    return y, tx

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    tx = np.ones(x.shape[0])
    for i in range(1,degree+1):
        tx = np.c_[tx,x**i]
    return tx

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

# Loss measurement 

def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred

def compute_loss(y, tx, w, error):
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
    else: raise NotImplemented # AMS

def cal_error(y, y_pred):
    """ Returns vector of 0,2 or -2, the difference between vector of labels and vector of predicted labels"""
    return y - y_pred
    
def cal_mse(e):
    """Returns the mean square error for vector e."""
    return 1/2*np.mean(e**2)

def cal_rmse(e):
    """Returns the root mean square error using the mean square error as input """
    return np.sqrt(2*cal_mse(e))

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
    
def accuracy(y,y_pred):
    """ Returns accuracy of classification = percentage of success"""
    return np.sum(y == y_pred)/len(y)

"""
def cal_loglike(y, tx, w):
    compute the cost by negative log likelihood.
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)
"""

def cal_loglike(y, tx, w): # A MODIFIER !!!
    """Compute the cost by negative log likelihood."""
    sigmo = sigmoid(tx.dot(w))

    #avoid issues zero log
    epsilon = 0
    if len(sigmo[(1-sigmo)==0]) > 0 or len(sigmo[sigmo==0])> 0:
        epsilon = 1e-9

    loss = y.T.dot(np.log(sigmo+epsilon)) + (1 - y).T.dot(np.log(1 - sigmo+epsilon))
    return np.squeeze(-loss)

def cal_loglike_r(y, tx, w, lambda_):
     return cal_loglike(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))

# Optimisation Methods 

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    w = initial_w
    
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, _ = compute_gradient(y, tx, w)
        # gradient w by descent update
        w = w - gamma * grad

    print("Gradient Descent (gamma = {gamma} ,{ti}): w ={weights}".format(gamma = gamma, ti=max_iters - 1,weights =w))     
    return w

def least_squares_SGD(y, tx, initial_w, max_iters, gamma, batchsize):
    """Stochastic gradient descent."""
    # Define parameters to store w and loss

    w = initial_w

    for n_iter in range(max_iters):
        for batch in range(batchsize):
            for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch, num_batches=1):
                # compute a stochastic gradient and loss
                grad, _ = compute_gradient(y_batch, tx_batch, w)
                # update w through the stochastic gradient update
                w = w - gamma * grad

        #print("SGD({bi}/{ti}): loss={l}".format(
        #      bi=n_iter, ti=max_iters - 1, l=loss))
    print("SGD(gamma = {gamma},{ti}): w={weight}".format(gamma=gamma,ti=max_iters - 1,weight=w))
    
    return w

def least_squares(y, tx):
    """calculate the least squares solution."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    
    print("Least squares: w={}".format(w))
    return w

def ridge_regression(y, tx, lambda_):
    """regularisation"""
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    
    #print("Ridge regression: w={}".format(w))
    return w


def sigmoid(t):
    return 1.0 / (1 + np.exp(-t))


def compute_logistic_gradient(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)
    return grad

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression with Gradient descent algorithm."""
    w = initial_w
    
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad = compute_logistic_gradient(y, tx, w)
        # gradient w by descent update
        w = w - gamma * grad

    print("Logistic regression: w={}".format(w))
    return w

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression with Gradient descent algorithm."""
    
    w = initial_w
    
    for n_iter in range(max_iters):
        #loss = compute_logistic_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
        gradient = compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w -= gamma * gradient
        
    print("Regularized logistic regression (lambda = {lamb},{ti}) : w={weights}".format(lamb = lambda_, ti=max_iters - 1,weights =w))   
    return w

# Cross val

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, degree, k, k_indices,method, error, hyperparams):
    """return the loss of ridge regression."""
    # get k'th subgroup in test, others in train
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    
    y_te = y[te_indice]
    y_tr = y[tr_indice]
    x_te = x[te_indice]
    x_tr = x[tr_indice]
    
    
    # form data with polynomial degree
    tx_tr = build_poly(x_tr, degree)
    tx_te = build_poly(x_te, degree)
        
    tx_tr, mean, std = standardize(tx_tr)
    tx_te, _, _ = standardize(tx_te, mean, std)
    
    #print('Mean and std of each feature in train set: {} , {}'.format(tx_tr.mean(axis = 0),tx_tr.std(axis = 0)))
    #print('Mean and std of each feature in test set: {} , {}'.format(tx_te.mean(axis = 0),tx_te.std(axis = 0)))
    
    if method == 'rr': w = ridge_regression(y_tr, tx_tr, hyperparams[0]) # ridge regression
    elif method == 'ls': w = least_squares(y_tr, tx_tr) # least square
    elif method == 'lsGD': w = least_squares_GD(y_tr, tx_tr, hyperparams[0], hyperparams[1], hyperparams[2]) # gradient descent
    elif method == 'lsSGD': w = least_squares_SGD(y_tr, tx_tr, hyperparams[0], hyperparams[1], hyperparams[2], hyperparams[3]) # stoch GD
    elif method == 'log': w = logistic_regression(y_tr, tx_tr, hyperparams[0], hyperparams[1], hyperparams[2]) # logistic reg
    elif method == 'rlog': w =reg_logistic_regression(y_tr, tx_tr, hyperparams[3], hyperparams[0], hyperparams[1], hyperparams[2]) # regularised logistic reg
    else: raise NotImplemented
    
   
    if method == 'log':
        loss_tr = cal_loglike(y_tr, tx_tr, w)
        loss_te = cal_loglike(y_te, tx_te, w)
    elif method == 'rlog': # A REVOIR SI CEST BON !!!!
        loss_tr = cal_loglike_r(y_tr, tx_tr, w, hyperparams[3])
        loss_te = cal_loglike_r(y_te, tx_te, w, hyperparams[3])
    else :
        # calculate the loss for train and test data
        loss_tr = compute_loss(y_tr, tx_tr, w, error)
        loss_te = compute_loss(y_te, tx_te, w, error)      
    
    
    y_pred = predict_labels(np.array(w).T, tx_te)
    acc = accuracy(y_te,y_pred)
    
    return loss_tr, loss_te, w, acc
     
    
def cross_validation_demo(y, x, degree, seed, k_fold = 4, class_distribution = False, error ='class', method='rr',hyperparams=[]):
    
    if class_distribution == True : y, x = equal_class(y,x)
    k_indices = build_k_indices(y, k_fold, seed)
           
    verify_proportion(y,k_indices)
    
    # cross validation
    loss_tr, loss_te, w, accuracy = choose_method(y, x, degree, seed, k_fold, k_indices, error, method, hyperparams)
    
        
    #cross_validation_visualization(hyperparams, loss_tr, loss_te) #A MODIFIER    
    return loss_tr, loss_te, w, accuracy

def cross_validation_demo_featselect(y, x, ranked_index, degree, seed, k_fold = 4, class_distribution = False, error ='class', method='rr',hyperparams=[]):
    
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
        loss_tr_tmp, loss_te_tmp, w_tmp, accuracy_tmp = choose_method(y, x_croped, degree, seed, k_fold, k_indices, error, method, hyperparams)
        loss_tr.append(loss_tr_tmp)
        loss_te.append(loss_te_tmp)
        w.append(w_tmp)
        accuracy.append(accuracy_tmp)
            
    
    #cross_validation_visualization(hyperparams, loss_tr, loss_te) #A MODIFIER    
    return loss_tr, loss_te, w, accuracy


def feat_augmentation(tx, threshold):
    
    corr_matrix = np.corrcoef(tx.T)
    index = np.argwhere(corr_matrix < threshold) #set threshold as an argument
    
    final_index = []
    for i in range(index.shape[0]):
        if index[i, 0] != index[i, 1]:
            final_index.append(index[i,:])
     
    final_index = np.sort(final_index, axis=1) 
    final_index = np.unique(final_index, axis=0)

    for i in range(final_index.shape[0]):
        feat1  = tx[:,final_index[i][0]]
        feat2  = tx[:,final_index[i][1]]
        tx = np.c_[tx, np.multiply(feat1,feat2)]


    
    return tx


def choose_method(y, x, degree, seed, k_fold = 4, k_indices = [], error ='class', method='rr',hyperparams=[]):
    
    loss_tr = []
    loss_te = []
    
    w = []
    accuracy = []
    
    if method == 'rr':
        for lambda_ in hyperparams[0]:
            loss_tr_tmp, loss_te_tmp, w_tmp, acc_tmp  = single_cross_val(y, x, degree, k_fold, k_indices, method,error,[lambda_])
            loss_tr.append(concate_fold(loss_tr_tmp))
            loss_te.append(concate_fold(loss_te_tmp))
            accuracy.append(acc_tmp)
            w.append(w_tmp)
    elif method == 'ls':
        loss_tr_tmp, loss_te_tmp, w_tmp, acc_tmp= single_cross_val(y, x, degree, k_fold, k_indices, method,error)
        loss_tr.append(concate_fold(loss_tr_tmp))
        loss_te.append(concate_fold(loss_te_tmp))
        accuracy.append(acc_tmp)
        w.append(w_tmp)
    elif method =='lsGD':
        for gamma in hyperparams[2]:
            loss_tr_tmp, loss_te_tmp, w_tmp, acc_tmp = single_cross_val(y, x, degree, k_fold, k_indices, method,error, [hyperparams[0],hyperparams[1],gamma])
            loss_tr.append(concate_fold(loss_tr_tmp))
            loss_te.append(concate_fold(loss_te_tmp))
            accuracy.append(acc_tmp)
            w.append(w_tmp)
    elif method =='lsSGD':
        for gamma in hyperparams[2]:
            loss_tr_tmp, loss_te_tmp, w_tmp, acc_tmp = single_cross_val(y, x, degree, k_fold, k_indices, method,error,[hyperparams[0],hyperparams[1],gamma,hyperparams[3]])
            loss_tr.append(concate_fold(loss_tr_tmp))
            loss_te.append(concate_fold(loss_te_tmp))
            accuracy.append(acc_tmp)
            w.append(w_tmp)
    elif method == 'log':
        for gamma in hyperparams[2]:
            loss_tr_tmp, loss_te_tmp, w_tmp, acc_tmp = single_cross_val(y, x, degree, k_fold, k_indices, method,error,[hyperparams[0],hyperparams[1],gamma])
            loss_tr.append(concate_fold(loss_tr_tmp))
            loss_te.append(concate_fold(loss_te_tmp))
            accuracy.append(acc_tmp)
            w.append(w_tmp)
    elif method == 'rlog':
        for lambda_ in hyperparams[3]:
            for gamma in hyperparams[2]:
                loss_tr_tmp, loss_te_tmp, w_tmp, acc_tmp = single_cross_val(y, x, degree, k_fold, k_indices, method,error,[hyperparams[0],hyperparams[1],gamma,lambda_])
            loss_tr.append(concate_fold(loss_tr_tmp))
            loss_te.append(concate_fold(loss_te_tmp))
            accuracy.append(acc_tmp)
            w.append(w_tmp)
    else: raise NotImplemented 
    return loss_tr, loss_te, w, accuracy
 

          
def single_cross_val(y, x, degree, k_fold, k_indices, method, error, hyperparams = []):
    loss_tr_tmp = []
    loss_te_tmp = []
    w_tmp = []
    accuracy = []
    
    for k in range(k_fold):
        loss_tr, loss_te, w , acc = cross_validation(y, x, degree, k, k_indices, method, error, hyperparams)
        loss_tr_tmp.append(loss_tr)
        loss_te_tmp.append(loss_te)    
        w_tmp.append(w)
        accuracy.append(acc)
    w_mean = np.mean(w_tmp,axis=0)
    #print("Accuracy = {}".format(accuracy))
    return loss_tr_tmp, loss_te_tmp, w_mean, accuracy

def equal_class(y,x):
    y_class0 = y[y==-1]
    y_class1 = y[y==1]
        
    x_class0 = x[y==-1][:]
    x_class1 = x[y==1][:]
        
    to_keep = np.random.permutation(len(y_class0))[:(len(y_class1)-1)]
    return  np.concatenate((y_class0[to_keep],y_class1),axis = 0), np.concatenate((x_class0[to_keep][:],x_class1),axis = 0)
          
def concate_fold(array_loss):
      #return np.mean(array_loss)
        return array_loss
    
def verify_proportion(y,k_indices):
    print('Number of remaining samples before start cross val : {}'.format(len(y)))
    print("Proportion of Bosons in all train set : {} %".format(100*len(y[y==1])/len(y)))
    print("Proportion of Bosons in test fold 1: {} %".format(100*len(y[k_indices[0]][y[k_indices[0]]==1])/len(y[k_indices[0]])))
    

# Visualisation methods 

def histo_visualization(feature_0,feature_1,index,std_number):
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    
    title_0 = 'Distribution of the features N°' + str(index+1)
    axes[0].set_title(title_0)
    axes[0].hist(feature_0,bins=100)
    axes[0].axvline(np.mean(feature_0), color='k', linewidth=1)
    axes[0].axvline( (std_number*np.std(feature_0)) + np.mean(feature_0), color='r', linestyle='dashed', linewidth=1)
    axes[0].axvline( (-std_number*np.std(feature_0)) + np.mean(feature_0), color='r', linestyle='dashed', linewidth=1)
    
    title_1 = 'Distribution of the features N°' + str(index+2)
    axes[1].set_title(title_1)
    axes[1].hist(feature_1,bins=100)
    axes[1].axvline(np.mean(feature_1), color='k', linewidth=1)
    axes[1].axvline( (std_number*np.std(feature_1)) + np.mean(feature_1), color='r', linestyle='dashed', linewidth=1)
    axes[1].axvline( (-std_number*np.std(feature_1)) + np.mean(feature_1), color='r', linestyle='dashed', linewidth=1)
    plt.show()

def scatter_visualization(label, feature_1,feature_2,index):
    fig = plt.figure()
   
    # plot 1
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.scatter(feature_1,label, marker=".", color='b', s=5)
    ax1.set_ylabel("Boson")
    ax1.set_xlabel("feature N°" + str(index+1))
    ax1.grid()

    # plot 2
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.scatter(feature_2, label, marker=".", color='b', s=5)
    ax2.set_ylabel("Boson")
    ax2.set_xlabel("feature N°" + str(index+2))
    ax2.grid()

    return fig
          

def cross_validation_visualization(lambds, loss_tr, loss_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, loss_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, loss_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")


def plot_correlation_matrix(tX, y, labels, figureName="CorrelationMatrix.png", threshold=0.85, print_correlated_pairs=False):
    """Computes and plots a heatmap of the correlation matrix. This matrix comprises the Pearson correlation
    coefficients between (continuous) features and the Point-biserial coefficients between each feature and 
    the (categorical) output."""
    
    correlation_output = [cal_point_biserial_correlation(tX[:,i], y) for i in range(tX.shape[1])]
    correlation_features = np.corrcoef(tX.T) 
    corr_matrix = np.c_[correlation_features, correlation_output]
    
    # Plot
    figure = plt.figure(figsize=(20,20))
    ax = figure.add_subplot(111)
    cax = ax.matshow(corr_matrix, cmap=plt.cm.PuOr)
    figure.colorbar(cax)
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)-1), labels[:-1])
    plt.tight_layout()
    plt.show()
    figure.savefig(figureName, bbox_inches='tight')
    
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
        
    return ranked_index, ranked_features

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


def result_crossval(loss_tr,loss_te,degree):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].boxplot(loss_tr)
    title_0 = "Train Errors of {deg} degree".format(deg=degree)
    axes[0].set_title(title_0)
    axes[0].set_ylabel('Error')
    axes[1].boxplot(loss_te)
    title_1 = "Test Errors of {deg} degree".format(deg=degree)
    axes[1].set_title(title_1)
    axes[1].set_ylabel('Error')
    plt.show()
    
def result_crossval_accuracy(acc,degree):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].boxplot(acc)
    title_0 = "Accuracy of {deg} degree".format(deg=degree)
    axes[0].set_title(title_0)
    axes[0].set_ylabel('Accuracy')   
    
    axes[1].plot(np.mean(acc,axis=1))
    title_1 = "Mean Accuracy of {deg} degree with".format(deg=degree)
    axes[1].set_title(title_1)
    axes[1].set_ylabel('Accuracy')
    plt.show()
    
def result_crossval_accuracy_feat(acc, lambdas):

    acc = np.mean(acc, axis=2)
    fig, axes = plt.subplots(1, 1, figsize=(12, 4))
    axes.plot(np.mean(acc,axis=1))
    axes.set_ylabel('Accuracy')
    axes.set_title('Mean accuracy as a function of feature number')
    best_lambdas = [lambdas[i] for i in np.argmax(acc, axis=1)]
    best_feature_set = np.argmax(np.mean(acc,axis=1))
    print("Best combination : {} features, lambda {}, acc. {}".format(best_feature_set+1, best_lambdas[best_feature_set], np.max(np.mean(acc,axis=1)))) # index 0 : 1 feature


def bias_variance_decomposition_visualization(degrees, loss_tr, loss_te):
    """visualize the bias variance decomposition."""
    
    loss_tr = np.array(loss_tr)
    loss_te = np.array(loss_te)
    
    tr_mean = np.expand_dims(np.mean(loss_tr, axis=0), axis=0)
    te_mean = np.expand_dims(np.mean(loss_te, axis=0), axis=0)
    plt.plot(degrees,loss_tr.T,'b',linestyle="-",color=([0.7, 0.7, 1]),label='train',linewidth=0.3)
    plt.plot(degrees,loss_te.T,'r',linestyle="-",color=[1, 0.7, 0.7],label='test',linewidth=0.3)
    plt.plot(degrees,tr_mean.T,'b',linestyle="-",label='train',linewidth=3)
    plt.plot(degrees,te_mean.T,'r',linestyle="-",label='test',linewidth=3)
    plt.ylim(0.2, 0.7)
    plt.xlabel("degree")
    plt.ylabel("error")
    plt.title("Bias-Variance Decomposition")
    plt.savefig("bias_variance")


