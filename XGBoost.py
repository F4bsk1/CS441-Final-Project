from ModelPredictor import ModelPredictor
import XGBoost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

class XGBoostPredictor(ModelPredictor):
    def __init__(self, hyperparameter_list=None):
        super().__init__("XGBoost", hyperparameter_list or {})
        #self.feature_cols = feature_cols
        #self.target_col = target_col

    def fit(self, train_X, train_y, hyperparameters = None):
        """the model params, target and features are specified when instanstiated
            simple hyperparam tuning are used (UPDATE LATER)"""
        #conver the dfs into DMatrixes for faster proccessing
        dtrain = xgb.DMatrix(train_X, label=train_y)


        #pass the params and train the model, use the built-in evaluation.
        cf = xgb.train(hyperparameters, 
                            dtrain,
                            num_boost_round=params.pop("num_boost_round", 500),
                            evals=[(dtrain, "train"), (dval, "val")],
                            early_stopping_rounds =params.pop("early_stopping_rounds", 20),
                            verbose_eval=False,
                            )
        return cf
    
    def predict(self, test_df):
        if self._model is None:
            raise RuntimeError("No model fitted, Call fit or run first")
        X_test = test_df[self.feature_cols].to_numpy()
        #predict 
        return self._model.predict(xgb.DMatrix(X_test))


    def evaluate(self, y_true, predictions):
        return np.sqrt(np.mean((y_true - predictions) ** 2)), r2_score(y_true, predictions)

    def tune_hyperparameters(self, train_X, train_y, val_X, val_y, hyperparameter_list):
        #simple grid search for hyperparameter tuning
        best_rmse = float("inf")
        best_params = {}
        for max_depth in hyperparameter_list.get("max_depth", [6]):
            for eta in hyperparameter_list.get("eta", [0.1]):
                params = {"objective": "reg:squarederror", "max_depth": max_depth, "eta": eta}
                dtrain = xgb.DMatrix(train_X, label=train_y)
                dval = xgb.DMatrix(val_X, label=val_y)
                cf = xgb.train(params, dtrain, num_boost_round=100, evals=[(dval, "val")], verbose_eval=False)
                preds = cf.predict(dval)
                rmse, _ = self.evaluate(val_y, preds)
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = params
        return best_params