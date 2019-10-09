#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 10:31:41 2019

@author: Juliane
"""

import numpy as np

def load_data(file_name="train.csv", sub_sample=True, add_outlier=False):
    """Load data."""
    path_dataset = file_name
    
    
    ID = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=0, dtype=int)
    
    converter = {1: lambda x: 0 if b"s" in x else 1}  #dummy coding

    label= np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=1, dtype=float, converters=converter)

    features = np.genfromtxt(path_dataset, delimiter=",", usecols=range(2,32), dtype=float, skip_header=1)

    
    # sub-sample
    if sub_sample:
        ID = ID[::50]
        label = label[::50]
        features = features[::50]

    #if add_outlier:
        # outlier experiment
        #height = np.concatenate([height, [1.1, 1.2]])
        #weight = np.concatenate([weight, [51.5/0.454, 55.3/0.454]])
    return ID, label, features

def build_model_data(features, label):
    """Form (y,tX) to get regression data in matrix form."""
    y = label
    x = features
    num_samples = len(y)
    tx = np.column_stack((x, np.ones(num_samples, dtype=x.dtype)))
    return y, tx


def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)



def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e = y - tx.dot(w)
    return calculate_mse(e)

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
    return w, loss


def least_squares(y, tx):
    """calculate the least squares solution."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    
    print("Least squares: loss={}".format(loss))
    print("Least squares: w={}".format(w))
    return w, loss




def ridge_regression(y, tx, lambda_): # = REGULARIZATION
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    
    print("Ridge regression: loss={}".format(loss))
    print("Ridge regression: w={}".format(w))
    return w, loss



def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0 / (1 + np.exp(-t))

def compute_logistic_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = sigmoid(tx.dot(w)) 
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)

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
    return w, loss




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
    return w, loss


