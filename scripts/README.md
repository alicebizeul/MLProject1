### CS-433: Machine learning - Project 1 ReadMe - BIZEUL Alice, DERVAUX Juliane, SIERRO Florent
### CS-433: Machine learning - Project 1 - ReadMe
### BIZEUL Alice, DERVAUX Juliane, SIERRO Florent
### Team: "BuzzLastyear" 


# run.py

Upload the data (training set and test set), which are in a "data" folder on the
The fourth cell allows the settings of the parameters for the model. For each subset, it is possible to fine-tune the hyperparameters:
	- The undefined features of DER_mass_MMC can be set to the median/mean of the feature, or even all the sample delete. 
	- Outliers of the train set can be removed while choosing the threshold value of the standardt deviation.
	- Augmentatin of feature, i.e. XiXj, can be activated or not.
	- Augmentatin of feature, e.g. XiXj, can be activated or not.
	- Final lambdas and degrees ofor the four models can be set up.
To have the output submission, just run the file, and a 'BuzzLastyear.csv' (i.e. the name of our team) will be create in the data folder.


# implementations.py

As requested the 6 function: 
	least_squares_GD(y, tx, initial w, max iters, gamma), least_squares_SGD(y, tx, initial w, max iters, gamma), least_squares(y, tx), ridge_regression(y, tx, lambda ), logistic_regression(y, tx, initial w, max iters, gamma), reg_logistic_regression(y, tx, lambda , initial w, max iters, gamma)
are in this single python file.


# helpers_*.py

Three helpers files (helpers_data.py, helpers_visualization.py, helpers_optimization.py) contains our own functions. All these files are import in the run.py
1. helpers_data.py:		Containst the implementations to load the data, split the data and engineering the features.
2. helpers_visualization.py: 	Containst the implementations to plot all our figures from our results such as histograms, boxplots, etc. 
3. helpers_optimization.py:	Containst the implementations to perform the optimisation of the different models, such as cross validation, computation of the loss, etc. 


## Jupiter Notbooks:

1. Exploratory data analysis.ipynb

This file explore the data set. It computes the number of samples, the number of features, look for NaN values and undefined. 
It plots some histograms to see the distributions of the features, to know if they are continuous or categorical, if they have lots of outliers.
It also plots the correlation matrix of each subset to see the correlation among the features.

2. SS*.ipynb

Here are 4 files (one for each subset) where the optimisation of the hyperparameters (i.e. lambda and degree) of our final model, which is ridge regression, are done.


3. least-square_ss*.ipynb

Here a 4 files testing the least squares method, to check for singular matrix