from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import root_mean_squared_error, r2_score
import pandas as pd

class ModelPredictor(ABC):
    """ Tobi please change to whetever is suitable from the streamlit UI preprocessing,
        Additional_params is a dictionary which we can pass specific model params"""
    def __init__(self, name, pred_horizon = 7):
        self.name = name
        self._fit_model = None #this will hold the model trained model object
        self.pred_horizon = pred_horizon

    # pass the 3 datasets into this method
    def find_best_params(self, train_df, val_df, test_df, hyperparameter_list, progress_callback=None):
        """All models follow this workflow of fit -> predict -> evaluate"""

        #train_df = self._create_lags(train_df)

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

        best_params, best_val_rmse = self.tune_hyperparameters(train_X, train_y, val_X, val_y, hyperparameter_list, progress_callback)
        #to be fixed
        #final_train = pd.concat([train_df, val_df], ignore_index=False)

        #final_train = self._create_lags(final_train)

        #final_train_X = final_train.drop(columns=['QUANTITY'])
        #final_train_y = final_train['QUANTITY']

        #self._fit_model = self.fit(final_train_X, final_train_y, best_params)
        #predictions = self.predict(test_X)
        #predictions = np.maximum(predictions, 0)  #no negative predictions
        #predictions = np.round(predictions)  #round to nearest integer
        #test_df = test_df.copy()
        #test_df = test_df.rename(columns={'QUANTITY': 'QUANTITY_TRUE'})
        #test_df['QUANTITY_PREDICTIONS'] = predictions

        #test_X['QUANTITY_PREDICTIONS'] = predictions
        #print("Test Data Samples with Predictions:")
        #print(test_X.head())
        #print(test_y.head())
        #test_X['QUANTITY_TRUE'] = test_y.values

        # Merge back by index (or by keys if you prefer)
        #test_out = test_X.join(preds_df)
        #test_out['QUANTITY_TRUE'] = test_y

        
        return best_params, best_val_rmse

    def run_on_test(self, train_df, val_df, test_df, best_params, progress_callback=None):
        test_X = test_df.drop(columns=['QUANTITY'])
        test_y = test_df['QUANTITY']

        final_train = pd.concat([train_df, val_df], ignore_index=False)
    
        final_train_X = final_train.drop(columns=['QUANTITY'])
        final_train_y = final_train['QUANTITY']

        self._fit_model = self.fit(final_train_X, final_train_y, best_params, progress_callback=progress_callback)
        predictions = self.predict(test_X)
        
        test_df = test_df.copy()
        test_df = test_df.rename(columns={'QUANTITY': 'QUANTITY_TRUE'})
        
        grouping = None
        if 'MAIN_GROUP' in test_df.columns:
            grouping = 'MAIN_GROUP'
        elif 'PRODUCT_GROUP' in test_df.columns:
            grouping = 'PRODUCT_GROUP'
        elif 'ARTICLE' in test_df.columns:
            grouping = 'ARTICLE'
        
        predictions['QUANTITY'] = np.maximum(predictions['QUANTITY'], 0)
        predictions['QUANTITY'] = np.round(predictions['QUANTITY'])
        predictions = predictions.rename(columns={'QUANTITY': 'QUANTITY_PREDICTIONS'})
        test_df = pd.merge(test_df, predictions, on=['DATE', grouping], how='left')
        pred_values = test_df['QUANTITY_PREDICTIONS'].fillna(0).values


        return test_df, self.evaluate(test_y, pred_values)

    def predict_future(self, train_df, val_df, test_df, best_params, category):
        final_train = pd.concat([train_df, val_df, test_df], ignore_index=False)

        final_train_X = final_train.drop(columns=['QUANTITY'])
        final_train_y = final_train['QUANTITY']

        self._fit_model = self.fit(final_train_X, final_train_y, best_params)
        begda = np.max(test_df['DATE'])
        future_dates = pd.date_range(start=begda + pd.Timedelta(days=1),
                                    periods=self.pred_horizon,
                                    freq='D')

        if 'MAIN_GROUP' in final_train_X.columns:
            grouping = 'MAIN_GROUP'
        elif 'PRODUCT_GROUP' in final_train_X.columns:
            grouping = 'PRODUCT_GROUP'
        elif 'ARTICLE' in final_train_X.columns:
            grouping = 'ARTICLE'
        else:
            raise ValueError("No valid grouping column found in training data.")

        categories = final_train_X[grouping].unique()

        pred_X = pd.DataFrame([
            {grouping: cat, 'DATE': date}
            for cat in categories
            for date in future_dates
        ])
        pred_X['DAY_OF_WEEK'] = pred_X['DATE'].dt.dayofweek


        predictions = self.predict(pred_X)
        #predictions = np.maximum(predictions, 0)  #no negative predictions
        #predictions = np.round(predictions)  #round to nearest integer

    @abstractmethod
    def fit(self, train_X, train_y, hyperparameters = None, progress_callback=None):
        pass
    
    @abstractmethod
    def predict(self, test_X):
        pass

    @abstractmethod
    def evaluate(self, y_true, predictions):
        pass #return root_mean_squared_error(y_true, predictions), r2_score(y_true, predictions)
    
    @abstractmethod
    def tune_hyperparameters(self, train_X, train_y, val_X, val_y, hyperparameter_list, progress_callback=None):
        pass

        
    


        
