#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 12:03:26 2019
@author: Juliane
"""

import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# Visualization methods 
# =============================================================================

def histo_visualization(feature_0,feature_1,index,std_number):
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    # plot 1
    title_0 = 'Distribution of the features N°' + str(index+1)
    axes[0].set_title(title_0)
    axes[0].hist(feature_0,bins=100)
    axes[0].axvline(np.mean(feature_0), color='k', linewidth=1)
    axes[0].axvline( (std_number*np.std(feature_0)) + np.mean(feature_0), color='r', linestyle='dashed', linewidth=1)
    axes[0].axvline( (-std_number*np.std(feature_0)) + np.mean(feature_0), color='r', linestyle='dashed', linewidth=1)
    # plot 2
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