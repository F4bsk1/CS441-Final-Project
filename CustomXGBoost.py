from ModelPredictor import ModelPredictor
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder


class XGBoostPredictor(ModelPredictor):
    def __init__(self, hyperparameter_list=None, pred_horizon = 7):
        super().__init__("XGBoost", hyperparameter_list or {})
        self.le = LabelEncoder()
        #self.feature_cols = feature_cols
        #self.target_col = target_col
        self.pred_horizon = pred_horizon

    def fit(self, train_X, train_y, hyperparameters = None, store_model = True):
        """the model params, target and features are specified when instanstiated
            simple hyperparam tuning are used (UPDATE LATER)"""
        #convert the dfs into DMatrixes for faster proccessing
        #dtrain = xgb.DMatrix(train_X, label=train_y)
        train_X = self._prep_dataset(train_X)

        model = XGBRegressor(
            n_estimators = hyperparameters.get("n_estimators", 100), 
            max_depth = hyperparameters.get("max_depth", 6),
            learning_rate = hyperparameters.get("learning_rate", 0.1),
            min_child_weight = hyperparameters.get("min_child_weight", 1),
            gamma = hyperparameters.get("gamma", 0),
            subsample = hyperparameters.get("subsample", 1),
            colsample_bytree = hyperparameters.get("colsample_bytree", 1),
            reg_alpha = hyperparameters.get("reg_alpha", 0),
            reg_lambda = hyperparameters.get("reg_lambda", 1),
            booster = hyperparameters.get("booster", "gbtree"),
            tree_method = hyperparameters.get("tree_method", "auto")
            )

        fit_model = model.fit(train_X.to_numpy(), train_y.to_numpy())
        if store_model:
            self._model = fit_model


        #pass the params and train the model, use the built-in evaluation.
        #cf = xgb.train(hyperparameters, dtrain, num_boost_round=100, evals=[(dval, "val")], verbose_eval=False)

        return fit_model
    
    def predict(self, test_X):
        if self._model is None:
            raise RuntimeError("No model fitted, call fit first")
        test_X_copy = self._prep_dataset(test_X)
        preds = self._model.predict(test_X_copy)
        test_X['QUANTITY'] = preds
        return test_X
        preds = []
        window = test_X.iloc[-1].to_numpy().reshape(1, -1)
        #for _ in range(n_steps):
        #    next_pred = self._model.predict(window)[0]
        #    preds.append(next_pred)
        #    #shift window by one step, append predicted value
        #    window = np.roll(window, -1)
        #    window[0, -1] = next_pred  #update last feature with prediction
        #return np.array(preds)

    def _prep_dataset(self, dataset):
        data_X = dataset.copy()
        if 'DATE' in data_X.columns:
            data_X = data_X.drop(columns=['DATE'])
        if 'QUANTITY' in data_X.columns:
            data_X = data_X.drop(columns=['QUANTITY'])
            
        if 'ARTICLE' in data_X.columns:
            data_X['ARTICLE'] = self.le.fit_transform(data_X['ARTICLE'])
        if 'PRODUCT_GROUP' in data_X.columns:
            data_X['PRODUCT_GROUP'] = self.le.fit_transform(data_X['PRODUCT_GROUP'])
        if 'MAIN_GROUP' in data_X.columns:
            data_X['MAIN_GROUP'] = self.le.fit_transform(data_X['MAIN_GROUP'])

        #print(data_X.head())
        return data_X
    

    def evaluate(self, y_true, predictions):
        return root_mean_squared_error(y_true, predictions), r2_score(y_true, predictions)

    def tune_hyperparameters(self, train_X, train_y, val_X, val_y, hyperparameter_list):
        #simple grid search for hyperparameter tuning
        val_transform_done = False

        best_rmse = float("inf")
        best_params = {}
        for max_depth in hyperparameter_list.get("max_depth", [6]):
            for learning_rate in hyperparameter_list.get("learning_rate", [0.1]):
                for n_estimators in hyperparameter_list.get("n_estimators", [100]):
                    for subsample in hyperparameter_list.get("subsample", [1]):
                        for colsample_bytree in hyperparameter_list.get("colsample_bytree", [1]):
                            for min_child_weight in hyperparameter_list.get("min_child_weight", [1]):
                                for gamma in hyperparameter_list.get("gamma", [0]):
                                    for reg_alpha in hyperparameter_list.get("reg_alpha", [0]):
                                        for reg_lambda in hyperparameter_list.get("reg_lambda", [1]):
                                            for booster in hyperparameter_list.get("booster", ['gbtree']):
                                                for tree_method in hyperparameter_list.get("tree_method", ['auto']):
                                                    params = {
                                                        "objective": "reg:squarederror",
                                                        "max_depth": max_depth,
                                                        "learning_rate": learning_rate,
                                                        "n_estimators": n_estimators,
                                                        "subsample": subsample,
                                                        "colsample_bytree": colsample_bytree,
                                                        "min_child_weight": min_child_weight,
                                                        "gamma": gamma,
                                                        "reg_alpha": reg_alpha,
                                                        "reg_lambda": reg_lambda,
                                                        "booster": booster,
                                                        "tree_method": tree_method}
                                                    #params = {"objective": "reg:squarederror", "max_depth": max_depth, "learning_rate": learning_rate}
                                                    #dtrain = xgb.DMatrix(train_X, label=train_y)
                                                    #dval = xgb.DMatrix(val_X, label=val_y)
                                                    #cf = xgb.train(params, dtrain, num_boost_round=100, evals=[(dval, "val")], verbose_eval=False)
                                                    #preds = cf.predict(dval)
                                                    clf = self.fit(train_X, train_y, params, store_model=False)
                                                    if not val_transform_done:
                                                        val_transform_done = True
                                                        val_X = self._prep_dataset(val_X)
                                                    #print("Val X after prep:")
                                                    #print(val_X.head())
                                                    preds = clf.predict(val_X)
                                                    rmse, _ = self.evaluate(val_y, preds)
                                                    print(f"Params: {params}, RMSE: {rmse}")
                                                    if rmse < best_rmse:
                                                        best_rmse = rmse
                                                        best_params = params
                                                        print(f"New best params found: {best_params} with RMSE: {best_rmse}")
        return best_params, best_rmse