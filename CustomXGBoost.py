from ModelPredictor import ModelPredictor
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder


class XGBoostPredictor(ModelPredictor):
    def __init__(self, hyperparameter_list=None):
        super().__init__("XGBoost", hyperparameter_list or {})
        self.le = LabelEncoder()
        #self.feature_cols = feature_cols
        #self.target_col = target_col

    def fit(self, train_X, train_y, hyperparameters = None, store_model = True):
        """the model params, target and features are specified when instanstiated
            simple hyperparam tuning are used (UPDATE LATER)"""
        #conver the dfs into DMatrixes for faster proccessing
        #dtrain = xgb.DMatrix(train_X, label=train_y)
        train_X = train_X.copy()
        train_X = train_X.drop(columns=['DATE'])
        train_X['ARTICLE'] = self.le.fit_transform(train_X['ARTICLE'])

        model = XGBRegressor(
            n_estimators = hyperparameters.get("n_estimators", 100), 
            max_depth = hyperparameters.get("max_depth", 6),
            learning_rate = hyperparameters.get("learning_rate", 0.1)
            )

        fit_model = model.fit(train_X.to_numpy(), train_y.to_numpy())
        if store_model:
            self._model = fit_model


        #pass the params and train the model, use the built-in evaluation.
        #cf = xgb.train(hyperparameters, dtrain, num_boost_round=100, evals=[(dval, "val")], verbose_eval=False)

        return fit_model
    
    def predict(self, test_X, n_steps=7):
        if self._model is None:
            raise RuntimeError("No model fitted, call fit first")
        test_X_copy = test_X.copy()
        test_X_copy = test_X_copy.drop(columns=['DATE'])
        test_X_copy['ARTICLE'] = self.le.transform(test_X_copy['ARTICLE'])
        preds = self._model.predict(test_X_copy)
        test_X['QUANTITY_PREDICTIONS'] = preds
        return preds
        preds = []
        window = test_X.iloc[-1].to_numpy().reshape(1, -1)
        #for _ in range(n_steps):
        #    next_pred = self._model.predict(window)[0]
        #    preds.append(next_pred)
        #    #shift window by one step, append predicted value
        #    window = np.roll(window, -1)
        #    window[0, -1] = next_pred  #update last feature with prediction
        #return np.array(preds)


    def evaluate(self, y_true, predictions):
        return root_mean_squared_error(y_true, predictions), r2_score(y_true, predictions)

    def tune_hyperparameters(self, train_X, train_y, val_X, val_y, hyperparameter_list):
        #simple grid search for hyperparameter tuning
        val_transfrom_done = False

        best_rmse = float("inf")
        best_params = {}
        for max_depth in hyperparameter_list.get("max_depth", [6]):
            for learning_rate in hyperparameter_list.get("learning_rate", [0.1]):
                params = {"objective": "reg:squarederror", "max_depth": max_depth, "learning_rate": learning_rate}
                #dtrain = xgb.DMatrix(train_X, label=train_y)
                #dval = xgb.DMatrix(val_X, label=val_y)
                #cf = xgb.train(params, dtrain, num_boost_round=100, evals=[(dval, "val")], verbose_eval=False)
                #preds = cf.predict(dval)
                clf = self.fit(train_X, train_y, params, store_model=False)
                if not val_transfrom_done:
                    val_transfrom_done = True
                    val_X = val_X.copy()
                    val_X = val_X.drop(columns=['DATE'])
                    val_X['ARTICLE'] = self.le.transform(val_X['ARTICLE'])
                preds = clf.predict(val_X)
                rmse, _ = self.evaluate(val_y, preds)
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = params
        return best_params