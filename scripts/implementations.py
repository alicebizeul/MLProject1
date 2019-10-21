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

def split_subsets(tX, y):
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
    
    ss0_tX, ss1_tX, ss2_tX, ss3_tX = remove_undef_feat(ss0_tX, ss1_tX, ss2_tX, ss3_tX)

    return ss0_tX, ss0_y, ss1_tX, ss1_y, ss2_tX, ss2_y, ss3_tX, ss3_y

def split_subsets_test(tX):
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
    
    ss0_tX, ss1_tX, ss2_tX, ss3_tX = remove_undef_feat(ss0_tX, ss1_tX, ss2_tX, ss3_tX)

    return ss0_tX, mask_ss0, ss1_tX, mask_ss1, ss2_tX, mask_ss2, ss3_tX, mask_ss3

def remove_undef_feat(ss0_tX, ss1_tX, ss2_tX, ss3_tX):
    
    #Now, we can remove the categorical feature "PRI_jet_num" for our subsets
    ss0_tX = np.delete(ss0_tX, 22, axis=1)
    ss1_tX = np.delete(ss1_tX, 22, axis=1)
    ss2_tX = np.delete(ss2_tX, 22, axis=1)
    ss3_tX = np.delete(ss3_tX, 22, axis=1)
    
    # Removing undefined features for the corresponding subsets
    features_undefined_ss01 = [4, 5, 6, 12, 25, 26, 27]
    ss0_tX = np.delete(ss0_tX, features_undefined_ss01, axis=1)
    ss1_tX = np.delete(ss1_tX, features_undefined_ss01, axis=1)
         
    features_undefined_ss0 = [18, 19, 20, 21] # taking into account indices of the features previously removed
    ss0_tX = np.delete(ss0_tX, features_undefined_ss0, axis=1)
    
    return ss0_tX, ss1_tX, ss2_tX, ss3_tX  

def replace_undef_feat(tX,method):
    if method == 'median' : tX[tX[:,0] == -999][0] = np.median(tX[~(tX[:,0] == -999)][0])
    elif method == 'mean' : tX[tX[:,0] == -999][0] = np.mean(tX[~(tX[:,0] == -999)][0])
    elif method == 'delete' : tX = np.delete(tX, tX[tX[:,0] == -999], 0) 
    return tX

def build_model_data(features, label):
    """Form (y,tX) to get regression data in matrix form."""
    y = label
    x = features
    num_samples = len(y)
    tx = np.column_stack((x, np.ones(num_samples, dtype=x.dtype)))
    return y, tx

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    tx = np.ones(x.shape[0])
    for i in range(1,degree+1):
        tx = np.c_[tx,x**i]
    return tx

def standardize(tx):
    tx = (tx - np.mean(tx, axis = 0))
    return np.c_[tx[:,0],tx[:,1:]/ np.std(tx[:,1:],axis = 0)]

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
    else: raise NotImplemented # AMS
        
def compute_logistic_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = sigmoid(tx.dot(w)) 
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)

def cal_error(y, y_pred):
    """ Returns vector of 0,2 or -2, the difference between vector of labels and vector of predicted labels"""
    return y - y_pred
    
def cal_mse(e):
    """Returns the mean square error for vector e."""
    return 1/2*np.mean(e**2)

def cal_rmse(mse):
    """Returns the root mean square error using the mean square error as input """
    return np.sqrt(2*mse)

def cal_classerror(y,y_pred):
    """Returns the class error (percentage of fails) which takes 
    reequilibrates class distribution into acount"""
    class1 = np.sum(y_pred[y ==1] != 1)/np.sum(y == 1)
    class2 = np.sum(y_pred[y == -1] != -1)/np.sum(y == -1)
    
    return class1 + class2

def cal_classificationerror(y, y_pred):
    """Returns the classification error = percentage of fails, does not 
    take class distribution among data set into account """
    return 1-accuracy(y,y_pred)
    
def accuracy(y,y_pred):
    """ Returns accuracy of classification = percentage of success"""
    return np.sum(y == y_pred)/len(y)

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
        grad, err = compute_gradient(y, tx, w)
        loss = calculate_mse(err)
        # gradient w by descent update
        w = w - gamma * grad
        
        print("Gradient Descent({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))
    print("Gradient Descent: w={}".format(w))   
    return w, loss

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from one example n and its corresponding y_n labels."""
    err = y - tx.dot(w)
    grad = -tx.T*err
    return grad, err

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent."""
    # Define parameters to store w and loss

    w = initial_w
    data_size = y.shape[0]
    
    for n_iter in range(max_iters):
        
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
        
        for n in range(data_size):
            # compute a stochastic gradient and loss
            grad, _ = compute_stoch_gradient(shuffled_y[n], shuffled_tx[n,:], w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = compute_loss(y, tx, w)

        print("SGD({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))
    print("SGD: w={}".format(w))
    return w

def least_squares(y, tx):
    """calculate the least squares solution."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    
    print("Least squares: loss={}".format(loss))
    print("Least squares: w={}".format(w))
    return w

def ridge_regression(y, tx, lambda_): # = REGULARIZATION
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    
    print("Ridge regression: w={}".format(w))
    return w

def sigmoid(t):
    """apply sigmoid function on t."""
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
        loss = compute_logistic_loss(y, tx, w)
        # gradient w by descent update
        w = w - gamma * grad
        
        print("Logistic regression: loss={}".format(loss))
    print("Logistic regression: w={}".format(w))
    return w

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression with Gradient descent algorithm."""
    
    w = initial_w
    
    for n_iter in range(max_iters):
        loss = compute_logistic_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
        gradient = compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w -= gamma * gradient
        print("Regularized logistic regression ({bi}/{ti}): loss={l}".format(
      bi=n_iter, ti=max_iters - 1, l=loss))
        
    print("Regularized logistic regression: w={}".format(w))   
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
        
    tx_tr = standardize(tx_tr)
    tx_te = standardize(tx_te)
    
    if method == 'rr': w = ridge_regression(y_tr, tx_tr, hyperparams[0]) # ridge regression
    elif method == 'ls': w = least_squares(y_tr, tx_tr) # least square
    elif method == 'lsGD': w = least_squares_GD(y_tr, tx_tr, hyperparams[0], hyperparams[1], hyperparams[2]) # gradient descent
    elif method == 'lsSGD': w = least_squares_SGD(y_tr, tx_tr, hyperparams[0], hyperparams[1], hyperparams[2]) # stoch GD
    elif method == 'log': w = logistic_regression(y_tr, tx_tr, hyperparams[0], hyperparams[1], hyperparams[2]) # logistic reg
    elif method == 'rlog': w =reg_logistic_regression(y_tr, tx_tr, hyperparams[3], hyperparams[0], hyperparams[1], hyperparams[2]) # regularised logistic reg
    else: raise NotImplemented
    
    if method == 'log' or method == 'rlog': # A REVOIR SI CEST BON !!!!
        loss_tr = compute_logistic_loss(y_tr, tx_tr, w)
        loss_te = compute_logistic_loss(y_te, tx_te, w)
    else :
        # calculate the loss for train and test data
        loss_tr = compute_loss(y_tr, tx_tr, w, error)
        loss_te = compute_loss(y_te, tx_te, w, error)      
    
    return loss_tr, loss_te, w

def cross_validation_demo(y, x, degree, seed, k_fold = 4, class_distribution = False, error ='class', method='rr',hyperparams=[]):
    
    if class_distribution == True : y, x = equal_class(y,x)
    k_indices = build_k_indices(y, k_fold, seed)
           
    verify_proportion(y,k_indices)

    loss_tr = []
    loss_te = []
          
    # cross validation
    if method == 'rr':
        for lambda_ in hyperparams[0]:
            loss_tr_tmp, loss_te_tmp = single_cross_val(y, x, degree, k_fold, k_indices, method,error,[lambda_])
            loss_tr.append(concate_fold(loss_tr_tmp)) # we could use something else then the mean
            loss_te.append(concate_fold(loss_te_tmp))
    elif method == 'ls':
        loss_tr_tmp, loss_te_tmp = single_cross_val(y, x, degree, k_fold, k_indices, method,error)
        loss_tr.append(concate_fold(loss_tr_tmp)) # we could use something else then the mean
        loss_te.append(concate_fold(loss_te_tmp))
    elif method =='lsGD':
        for gamma in hyperparams[2]:
            loss_tr_tmp, loss_te_tmp = single_cross_val(y, x, degree, k_fold, k_indices, method,error,[hyperparams[0],hyperparams[1],gamma])
            loss_tr.append(concate_fold(loss_tr_tmp)) # we could use something else then the mean
            loss_te.append(concate_fold(loss_te_tmp))
    elif method =='lsSGD':
        for gamma in hyperparams[2]:
            loss_tr_tmp, loss_te_tmp = single_cross_val(y, x, degree, k_fold, k_indices, method,error,[hyperparams[0],hyperparams[1],gamma])
            loss_tr.append(concate_fold(loss_tr_tmp)) # we could use something else then the mean
            loss_te.append(concate_fold(loss_te_tmp))
    elif method == 'log':
        for gamma in hyperparams[2]:
            loss_tr_tmp, loss_te_tmp = single_cross_val(y, x, degree, k_fold, k_indices, method,error,[hyperparams[0],hyperparams[1],gamma])
            loss_tr.append(concate_fold(loss_tr_tmp)) # we could use something else then the mean
            loss_te.append(concate_fold(loss_te_tmp))
    elif method == 'rlog':
        for lambda_ in hyperparams[3]:
            for gamma in hyperparams[2]:
                loss_tr_tmp, loss_te_tmp = single_cross_val(y, x, degree, k_fold, k_indices, method,error,[hyperparams[0],hyperparams[1],gamma,lambda_])
            loss_tr.append(concate_fold(loss_tr_tmp)) # we could use something else then the mean
            loss_te.append(concate_fold(loss_te_tmp))                
    else: raise NotImplemented  
          
    #cross_validation_visualization(hyperparams, loss_tr, loss_te) #A MODIFIER 

          
def single_cross_val(y, x, degree, k_fold, k_indices, method, error, hyperparams = []):
    loss_tr_tmp = []
    loss_te_tmp = []
    
    for k in range(k_fold):
        loss_tr, loss_te,_ = cross_validation(y, x, degree, k, k_indices, method, error, hyperparams)
        loss_tr_tmp.append(loss_tr)
        loss_te_tmp.append(loss_te)
    return loss_tr_tmp, loss_te_tmp

def equal_class(y,x):
    y_class0 = y[y==-1]
    y_class1 = y[y==1]
        
    x_class0 = x[y==-1][:]
    x_class1 = x[y==1][:]
        
    to_keep = np.random.permutation(len(y_class0))[:(len(y_class1)-1)]
    return  np.concatenate((y_class0[to_keep],y_class1),axis = 0), np.concatenate((x_class0[to_keep][:],x_class1),axis = 0)
          
def concate_fold(array_loss):
      return np.mean(array_loss)
    
def verify_proportion(y,k_indices):
    print('Number of remaining samples before start cross val : {} %'.format(len(y)))
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


def plot_correlation_matrix(tX, y, labels, figureName="CorrelationMatrix.png"):
    
    full_data = np.c_[tX, y]
    corr_matrix = np.corrcoef(full_data.T) 

    figure = plt.figure(figsize=(20,20))
    ax = figure.add_subplot(111)
    cax = ax.matshow(corr_matrix, cmap=plt.cm.PuOr)
    figure.colorbar(cax)
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.tight_layout()
    plt.show()
    figure.savefig(figureName, bbox_inches='tight')
    
    
    output_corr = corr_matrix[:,-1]
    output_corr = np.abs(output_corr[:-1])
    ranked_index = output_corr.argsort()
    ranked_features = [labels[i] for i in ranked_index]
    print("Ranked absolute correlation with output: ", np.sort(output_corr))
    print("Ranked features: ", ranked_features)
    return ranked_index, ranked_features


def outliers_suppresion(subsample, std_number):

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
    print("size of the dataset with {in_} and without {out} the outliers".format(in_=subsample.shape, out=subsample_outliers.shape))
    print("Number of sample suppressed ouside {std} std: {supp}".format(std=std_number, supp=(subsample.shape[0] - subsample_outliers.shape[0])))

    return subsample_outliers
