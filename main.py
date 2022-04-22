#!/usr/bin/env python3
# -*- coding: utf-8 -*-
''' 
Test for Titannic survival model optimalization, call python main.py [n samples] 
In line 15 set optimalization method [randomized, bayes, hyperband]
'''
import sys
import logging
from catboost.datasets import titanic
from random import randint

from titannic import SurvivalPrediction
from optimizers import Optimalization

opt_method='randomized' #randomized bayes hyperband

if __name__ == '__main__':
    
    data, _ = titanic()
    
    if len(sys.argv) > 1:
        sample_len = int(sys.argv[1])
    else: 
        sample_len = 6
    
    sample_start = randint(0, len(data)-sample_len)
    sample = data.iloc[sample_start:(sample_start+sample_len)]
    
    prediction = SurvivalPrediction()
    
    x, y, x_train, x_test, y_train, y_test, cat_features_index = prediction.prepare_data()
    logging.info('Data prepared') 
    
    optimalization = Optimalization()
    method_choosen = optimalization.optimalization_method(method=opt_method) 
    
    model_opt = prediction.train_model(method_choosen(prediction.prepare_model()),
 	                                   x_train, 
 	                                   x_test, 
 	                                   y_train, 
 	                                   y_test, 
 	                                   cat_features_index
 	                                   )
    if model_opt:
        logging.info('Model trained & optimized') 
    
    print(model_opt.get_params()) 
    print(model_opt.best_score_)
    
    prediction.cross_validation(model_opt, 
                                x, 
                                y, 
                                cat_features_index
                                )
    prediction.predict(model_opt, sample)
