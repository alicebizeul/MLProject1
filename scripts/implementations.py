#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 10:31:41 2019
@author: Juliane
"""

import numpy as np
import matplotlib.pyplot as plt
 
# =============================================================================
# Optimization Methods 
# =============================================================================


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    
    from helpers_optimization import compute_gradient, compute_loss
    
    w = initial_w
    
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, _ = compute_gradient(y, tx, w)
        # gradient w by descent update
        w = w - gamma * grad
        
    print("Gradient Descent (gamma = {gamma} ,{ti}): w ={weights}".format(gamma = gamma, ti=max_iters - 1,weights =w))    
    loss = compute_loss(y, tx, w, 'rmse')
    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma, batchsize):
    """Stochastic gradient descent."""
    
    from helpers_data import batch_iter
    from helpers_optimization import compute_gradient, compute_loss
    
    w = initial_w

    for n_iter in range(max_iters):
        for batch in range(batchsize):
            for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch, num_batches=1):
                # compute a stochastic gradient and loss
                grad, _ = compute_gradient(y_batch, tx_batch, w)
                # update w through the stochastic gradient update
                w = w - gamma * grad

    print("SGD(gamma = {gamma},{ti}): w={weight}".format(gamma=gamma,ti=max_iters - 1,weight=w))
    loss = compute_loss(y, tx, w, 'rmse')
    return w, loss


def least_squares(y, tx):
    """Least squares optimisation. Returns the optimal weights vector"""
    
    from helpers_optimization import compute_loss
    
    weights = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))

    loss = compute_loss(y, tx, weights, 'rmse')
    return weights, loss


def ridge_regression(y, tx, lambda_):
    """Least squares with regularisation. Returns optimal weight vector"""
    
    from helpers_optimization import compute_loss
    
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    weights = np.linalg.solve(tx.T.dot(tx) + aI, tx.T.dot(y))
    
    loss = compute_loss(y, tx, weights, 'rmse')
    return weights, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression with Gradient descent algorithm."""
    
    from helpers_optimization import compute_logistic_gradient, compute_loss
    
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad = compute_logistic_gradient(y, tx, w)
        # gradient w by descent update
        w = w - gamma * grad

    print("Logistic regression: w={}".format(w))
    loss = compute_loss(y, tx, w, 'logl')
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression with Gradient descent algorithm."""
    
    from helpers_optimization import compute_logistic_gradient, compute_loss
    
    w = initial_w
    for n_iter in range(max_iters):
        #loss = compute_logistic_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
        gradient = compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w -= gamma * gradient
        
    print("Regularized logistic regression (lambda = {lamb},{ti}) : w={weights}".format(lamb = lambda_, ti=max_iters - 1,weights =w))
    loss = compute_loss(y, tx, w, 'logl_r')
    return w, loss



