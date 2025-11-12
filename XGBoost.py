from ModelPredictor import ModelPredictor
import XGBoost as xgb
import numpy as np
import pandas as pd

class XGBoostPredictor(ModelPredictor):
    def __init__(self, name, time_window, granularity, feature_cols, target_col, additional_params=None):
        super().__init__("XGBoost", time_window, granularity, additional_params or {})
        self.feature_cols = feature_cols
        self.target_col = target_col

    def fit(self, train_df, val_df):
        """the model params, target and features are specified when instanstiated
            simple hyperparam tuning are used (UPDATE LATER)"""
        X_train = train_df[self.feature_cols].to_numpy()
        y_train = train_df[self.target_col].to_numpy()
        X_val = val_df[self.feature_cols].to_numpy()
        y_val = val_df[self.target_col].to_numpy()
        #conver the dfs into DMatrixes for faster proccessing
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        #set default parameters and let the instanciation override the default params
        default_params = {"objective": "reg:squarederror", "max_depth": 6, "eta": 0.1} #update later
        params = {**default_params, **(self.additional_params or {})}
        #pass the params and train the model, use the built-in evaluation.
        booster = xgb.train(params, 
                            dtrain,
                            num_boost_round=params.pop("num_boost_round", 500),
                            evals=[(dtrain, "train"), (dval, "val")],
                            early_stopping_rounds =params.pop("early_stopping_rounds", 20),
                            verbose_eval=False,
                            )
        return booster
    
    def predict(self, test_df):
        if self._model is None:
            raise RuntimeError("No model fitted, Call fit or run first")
        X_test = test_df[self.feature_cols].to_numpy()
        #predict 
        return self._model.predict(xgb.DMatrix(X_test))



