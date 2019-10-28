### CS-433: Machine learning - Project 1 ReadMe - BIZEUL Alice, DERVAUX Juliane, SIERRO Florent


# run.py

Upload the data (training set and test set), which are in a "data" folder on the repository of the run.py file.
The fourth cell allows the settings of the parameters for the model. For each subset, it is possible to tune the hyperparameters:
	- The undefined features of DER_mass_MMC can be set to the median/mean of the feature, or even all the sample delete. 
	- Outliers of the train set can be removed while choosing the threshold value of the standardt deviation.
	- Augmentatin of feature, i.e. XiXj, can be activated or not.
	- Final lambdas and degrees ofor the four models can be set up.
To have the output submission, just run the file, and a 'BuzzLastyear.csv' (i.e. the name of our team) will be create in the data folder.


# implementations.py




## Jupiter Notbooks:

1. Exploratory data analysis.ipynb

This file explore the data set. It computes the number of samples, the number of features, look for NaN values and undefined. 
It plots some histograms to see the distributions of the features, to know if they are continuous or categorical, if they have lots of outliers.
It also plots the correlation matrix of each subset to see the correlation among the features.

2. SS*.ipynb


3. least-square_ss*.ipynb

