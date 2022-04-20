#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#######################

import logging
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from hpbandster_sklearn import HpBandSterSearchCV
import ConfigSpace.hyperparameters as CSH
import ConfigSpace as CS


class Optimalization():
    ''' 
    Various ML optimalization kinds 
    '''
    def prepare_randomized_opt(self, model):
        ''' 
        Randomized optimizer preparing 
        '''
        param_distributions={'learning_rate': np.power(10, np.linspace(-2, -0.25, 10)),
                             'depth': np.arange(2, 9) },
        optimizer = RandomizedSearchCV(
                                model,
                                param_distributions,
                                cv=3,
                                n_jobs=3, 
                                n_iter=100,
                                return_train_score=False,
                                error_score='raise'
                                ) 
        if optimizer:
            logging.info('Randomized optimalization choosen') 
        return optimizer

    def prepare_bayes_opt(self, model):
        ''' 
        Bayes optimizer preparing 
        '''
        search_spaces=dict(learning_rate=Real(0.01, 0.6, prior='log-uniform'),
                           depth=Integer(2, 8))
        optimizer = BayesSearchCV(model,
                                  search_spaces,
                                  cv=3,
                                  n_jobs=3, 
                                  n_iter=100,
                                  return_train_score=False,
                                  error_score='raise'
                                  ) 
        if optimizer:
            logging.info('Bayes optimalization choosen') 
        return optimizer
    
    def prepare_hyperband_opt(self, model):
        ''' 
        Hyperband & BOHB optimizer preparing 
        '''
        param_distributions = CS.ConfigurationSpace()
        param_distributions.add_hyperparameter(CSH.UniformFloatHyperparameter("learning_rate", 
                                                                              lower=0.01, 
                                                                              upper=0.6, 
                                                                              log=True
                                                                              )
                                               )
        param_distributions.add_hyperparameter(CSH.UniformIntegerHyperparameter("depth", 2, 8))
        
        optimizer = HpBandSterSearchCV(model,
                                       param_distributions,
                                       cv=3,
                                       n_jobs=3, 
                                       n_iter=100, 
                                       verbose=0,
                                       optimizer = 'bohb' #hyperband bohb
                                       )
        if optimizer:
            logging.info('Hyperband optimalization choosen')
        return optimizer
    
    def optimalization_method(self, method='bayes'):
        ''' 
        Choose optimalization way 
        '''
        if method=='randomized':
            optimizer =  self.prepare_randomized_opt
        elif method=='bayes':
            optimizer =  self.prepare_bayes_opt
        elif method=='hyperband':
            optimizer =  self.prepare_hyperband_opt
        else:
            optimizer =  self.prepare_bayes_opt
        return optimizer

        ################