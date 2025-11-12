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
        print("Training Data Samples:")
        print("----------------------")
        print("train_X samples:")
        print(train_X.head())
        print("train_y samples:")
        print(train_y.head())

        val_X = val_df.drop(columns=['QUANTITY'])
        val_y = val_df['QUANTITY']

        test_X = test_df.drop(columns=['QUANTITY'])
        test_y = test_df['QUANTITY']

        best_params = self.tune_hyperparameters(train_X, train_y, val_X, val_y, self.hyperparameter_list)
        
        final_train = pd.concat([train_df, val_df], ignore_index=True)
        final_train_X = final_train.drop(columns=['QUANTITY'])
        final_train_y = final_train['QUANTITY']

        self._fit_model = self.fit(final_train_X, final_train_y, best_params)
        predictions = self.predict(test_X)
        predictions = np.maximum(predictions, 0)  #no negative predictions
        predictions = np.round(predictions)  #round to nearest integer
        test_X['QUANTITY_PREDICTIONS'] = predictions
        test_X['QUANTITY_TRUE'] = test_y.values
        return test_X

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



        
    


        
