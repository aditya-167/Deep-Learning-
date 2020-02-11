# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 23:43:04 2020

@author: Aditya Srichandan
"""
import numpy as np
import pandas as pd

class Data_prep:
    def read_file(self,string):
        sample=pd.read_csv(string)
        return sample
    
    def clean_data(self,sample):
        dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
        for each in dummy_fields:
            dummies = pd.get_dummies(sample[each], prefix=each, drop_first=False)
            sample = pd.concat([sample, dummies], axis=1)
    
        fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 'weekday', 'atemp', 'mnth', 'workingday', 'hr']
        data = sample.drop(fields_to_drop, axis=1)
        return data
    
    def standardize_data(self,data):
        quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
        scaled_features = {}
        for each in quant_features:
            mean, std = data[each].mean(), data[each].std()
            scaled_features[each] = [mean, std]
            data.loc[:, each] = (data[each] - mean)/std
        return data
    def split_data(self,data):
        fake = np.random.choice(data.index, size=int(len(data)*0.8), replace=False)
        data, test_data = data.iloc[fake], data.drop(fake)
        target_fields=['cnt','casual','registered']
        features, targets = data.drop(target_fields, axis=1), data[target_fields]
        features_test, targets_test = test_data.drop(target_fields, axis=1), test_data[target_fields]
        return features,targets,features_test,targets_test
    def head(self,data):
        print(data.head())

