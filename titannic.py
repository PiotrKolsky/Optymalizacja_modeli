#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#######################

import logging
import numpy as np
import pandas as pd
from catboost.datasets import titanic
from catboost import CatBoostClassifier, Pool, cv
from sklearn.model_selection import train_test_split


class SurvivalPrediction():
    ''' 
    Catboost Titannic survival set prediction example, just for model optimalization 
    '''
    def __init__(self):
        logging.basicConfig(level=logging.INFO)

    def prepare_data(self):
        ''' 
        Data preparing 
        '''
        titanic_train, _ = titanic()
        titanic_train.fillna(-999, inplace=True)

        x = titanic_train.drop(['Survived', 'PassengerId'], axis=1)
        y = titanic_train.Survived

        x_train, x_test, y_train, y_test = train_test_split(x, 
                                                            y, 
                                                            train_size=.85, 
                                                            )
        cat_features_index = np.where(x_train.dtypes != float)[0]
        
        return x, y, x_train, x_test, y_train, y_test, cat_features_index

    def prepare_model(self):
        ''' 
        ML engine preparing 
        '''
        model = CatBoostClassifier(custom_metric='Accuracy',
                                   loss_function='Logloss',
                                   use_best_model=True,
                                   n_estimators=50
                                   )
        if model:
            logging.info('Model prepared')  
        return model
    
    def train_model(self, 
                    optimizer, 
                    x_train, 
                    x_test, 
                    y_train, 
                    y_test, 
                    cat_features_index
                    ):
        '''
        Model training & optimizing
        '''
        optimizer.fit(x_train, 
                      y_train, 
                      cat_features=cat_features_index, 
                      eval_set=(x_test, y_test), 
                      verbose=False
                      )
        return optimizer.best_estimator_
 
    def cross_validation(self, 
                         model, 
                         x, 
                         y, 
                         cat_features_index
                         ):
        '''
        Accuracy cv counting
        '''
        cv_data = cv(Pool(x, 
                          y, 
                          cat_features=cat_features_index
                          ), 
                     model.get_params(), 
                     iterations=10, 
                     fold_count=3, 
                     verbose=False
                     )
        print('The best train cv accuracy: {:1.3f}'.format(np.max(cv_data["train-Accuracy-mean"])))
        print('The best test cv accuracy: {:1.3f}'.format(np.max(cv_data["test-Accuracy-mean"])))

    def predict(self, 
                model, 
                df
                ):
        '''
        Survival probability predicting
        '''
        x = df.drop(['PassengerId', 'Survived'], axis=1)
        y = df.Survived
        
        pred = pd.DataFrame([y.tolist(), model.predict_proba(x)[:, 1]]).T
        pred.columns=['Survived', 'Pred_Proba']
        pred['Survived'] = pred['Survived'].astype('int')
        print( pred.round({'Pred_Proba': 2}))

        #####################################