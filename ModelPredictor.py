from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import root_mean_squared_error, r2_score
import pandas as pd

class ModelPredictor(ABC):
    """ Tobi please change to whetever is suitable from the streamlit UI preprocessing,
        Additional_params is a dictionary which we can pass specific model params"""
    def __init__(self, name, hyperparameter_list=None):
        self.name = name
        self.hyperparameter_list = hyperparameter_list
        self._fit_model = None #this will hold the model trained model object

    # pass the 3 datasets into this method
    def run(self, train_df, val_df, test_df):
        """All models follow this workflow of fit -> predict -> evaluate"""
        train_X = train_df.drop(columns=['QUANTITY'])
        train_y = train_df['QUANTITY']

        val_X = val_df.drop(columns=['QUANTITY'])
        val_y = val_df['QUANTITY']

        test_X = test_df.drop(columns=['QUANTITY'])
        test_y = test_df['QUANTITY']

        best_params = self.model.tune_hyperparameters(train_X, train_y, val_X, val_y, self.hyperparameter_list)
        
        final_train = pd.concat([train_df, val_df], ignore_index=True)
        final_train_X = final_train.drop(columns=['QUANTITY'])
        final_train_y = final_train['QUANTITY']

        self._fit_model = self.model.fit(final_train_X, final_train_y, best_params)
        predictions = self._fit_model.predict(test_X)
        return self._fit_model.evaluate(test_y, predictions)

    @abstractmethod
    def fit(self, train_X, train_y, hyperparameters = None):
        pass
    
    @abstractmethod
    def predict(self, test_X):
        pass

    @abstractmethod
    def evaluate(self, y_true, predictions):
        pass #return root_mean_squared_error(y_true, predictions), r2_score(y_true, predictions)
    
    @abstractmethod
    def tune_hyperparameters(self, train_X, train_y, val_X, val_y, hyperparameter_list):
        pass



        
    


        
