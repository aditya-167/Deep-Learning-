# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 23:43:04 2020

@author: Aditya Srichandan
"""
import numpy as np
import pandas as pd
from Data_cleaning import Data_prep
import my_network 
def MSE(y,Y):
    return np.mean((y-Y)**2)
def Data(some_path):
    Reader=Data_prep()
    data=Reader.read_file(path)
    data=Reader.clean_data(data)
    data=Reader.standardize_data(data)
    return data

path='Bike-Sharing-Dataset/hour.csv'
data=Data(path)
X_train,Y_train,Y_test,Y_test=Reader.split_data(data)

N = X_train.shape[1]
network = NeuralNetwork(N, hidden_nodes, output_nodes, learning_rate)

losses = {'train':[], 'validation':[]}
for ii in range(iterations):
    batch = np.random.choice(X_train.index, size=256)
    X, y = X_train.ix[batch].values, Y_train.ix[batch]['cnt']
    network.train(X, y)
    train_loss = MSE(network.run(X_train).T, Y_train['cnt'].values)
    test_loss = MSE(network.run(X_test).T, Y_test['cnt'].values)
    losses['train'].append(train_loss)
    losses['validation'].append(test_loss)


plt.plot(losses['train'], label='Training loss')
plt.plot(losses['validation'], label='Validation loss')
plt.legend()
_ = plt.ylim()
