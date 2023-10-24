# NFL-ML-Playground
A collection of Machine Learning algorithms to predict play success in the NFL

Data was obtained from Kaggle here - https://www.kaggle.com/datasets/maxhorowitz/nflplaybyplay2009to2016
v5 of the dataset (2009 - 2018) was used.

This repository includes all data preprocessing. Complete play (cp) was scraped together as well with the intention of trying to predict play type
given down, distance to go, distance to goal line, time remaining, score differential, and a few other datapoints.
This proved to be fairly challenging, as there ended up being a total of 13 different play types, so accuracy was low given how random
events and playcalling in football can be.

Future investigation may call for predicting play types in specific scenarios, i.e. first offensive drive of the game for a team
given that these first few plays are often scripted during the week of preparation.

Play success was estimated using Scikit-Learn's Decision Tree Classifier, Random Forest Classifier, K-Nearest Neigbors,
and a Multi-Layer Perceptron (MLP).

Play success was also estimated using a standard feedforward fully connected neural network from PyTorch.

Both the Scikit MLP and the PyTorch NN achieved around 70% accuracy.
