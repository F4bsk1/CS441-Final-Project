from ModelPredictor import ModelPredictor
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import root_mean_squared_error, r2_score

class SARIMAPredictor(ModelPredictor):
    def __init__(self, hyperparameter_list=None, pred_horizon = 7):
        """
        hyperparameter_list example:
        {
            "order": (p, d, q),
            "seasonal_order": (P, D, Q, s)
        }
        """
        super().__init__("SARIMA", hyperparameter_list or {})
        self.models_per_category = {}  # store SARIMA model per category
        self.pred_horizon = pred_horizon  # default forecast horizon

    def fit(self, train_X, train_y, hyperparameters=None, store_model=True):
        """
        Fit SARIMA per category.
        Assumes 'CATEGORY' column exists in train_X.
        """
        train_X_copy = train_X.copy()
        train_y_copy = train_y.copy()

        if 'MAIN_GROUP' in train_X_copy.columns:
            grouping = 'MAIN_GROUP'
        elif 'PRODUCT_GROUP' in train_X_copy.columns:
            grouping = 'PRODUCT_GROUP'
        elif 'ARTICLE' in train_X_copy.columns:
            grouping = 'ARTICLE'
        else:
            raise ValueError("No valid grouping column found in train_X.")

        order = (1, 1, 0)
        seasonal_order = (0, 1, 1, 7)  # default weekly seasonality

        if hyperparameters:
            order = hyperparameters.get("order", order)
            seasonal_order = hyperparameters.get("seasonal_order", seasonal_order)

        categories = train_X_copy[grouping].unique()

        for cat in categories:
            cat_y = train_y_copy[train_X_copy[grouping] == cat]
            print("Fitting SARIMA for category:", cat)
            print(cat_y)
            model = SARIMAX(cat_y, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
            fit_model = model.fit(disp=False)

            if store_model:
                self.models_per_category[cat] = fit_model

        return self.models_per_category

    def predict(self, test_X):
        """
        Predict next n days per category.
        """
        preds = np.zeros(len(test_X))
        test_X_copy = test_X.copy()
        if 'MAIN_GROUP' in test_X_copy.columns:
            grouping = 'MAIN_GROUP'
        elif 'PRODUCT_GROUP' in test_X_copy.columns:
            grouping = 'PRODUCT_GROUP'
        elif 'ARTICLE' in test_X_copy.columns:
            grouping = 'ARTICLE'
        else:
            raise ValueError("No valid grouping column found in test_X.")

        for cat, model in self.models_per_category.items():
            cat_idx = test_X_copy[grouping] == cat
            if cat_idx.sum() == 0:
                continue

            forecast = model.get_forecast(steps=self.pred_horizon).predicted_mean

            n = cat_idx.sum()
            preds[cat_idx] = forecast[:n]

        return preds

    def evaluate(self, y_true, predictions):
        """
        Evaluate RMSE and R²
        """
        rmse = root_mean_squared_error(y_true, predictions)
        r2 = r2_score(y_true, predictions)
        return rmse, r2

    def tune_hyperparameters(self, train_X, train_y, val_X, val_y, hyperparameter_list):
        """
        Simple grid search for SARIMA parameters.
        hyperparameter_list should include:
        - p, d, q for ARIMA order
        - P, D, Q, s for seasonal order
        """
        best_rmse = float("inf")
        best_params = {
            "order": (1, 1, 0),
            "seasonal_order": (0, 1, 1, 7)
        }

        if 'MAIN_GROUP' in train_X.columns:
            grouping = 'MAIN_GROUP'
        elif 'PRODUCT_GROUP' in train_X.columns:
            grouping = 'PRODUCT_GROUP'
        elif 'ARTICLE' in train_X.columns:
            grouping = 'ARTICLE'
        else:
            raise ValueError("No valid grouping column found in test_X.")

        p_values = hyperparameter_list.get("p", [0, 1])
        d_values = hyperparameter_list.get("d", [0, 1])
        q_values = hyperparameter_list.get("q", [0, 1])

        P_values = hyperparameter_list.get("P", [0, 1])
        D_values = hyperparameter_list.get("D", [0, 1])
        Q_values = hyperparameter_list.get("Q", [0, 1])
        s_values = hyperparameter_list.get("s", [7])  # weekly seasonality

        categories = train_X[grouping].unique()

        for p in p_values:
            for d in d_values:
                for q in q_values:
                    for P in P_values:
                        for D in D_values:
                            for Q in Q_values:
                                for s in s_values:
                                    current_order = (p, d, q)
                                    current_seasonal = (P, D, Q, s)
                                    rmses = []
                                    print(f"Testing SARIMA order: {current_order}, seasonal_order: {current_seasonal}")

                                    for cat in categories:
                                        train_cat_y = train_y[train_X[grouping] == cat]
                                        val_cat_y = val_y[val_X[grouping] == cat]
                                        print("Fitting SARIMA for category:", cat)
                                        #print(f"train cat y: {train_cat_y}")
                                        #print(f"val cat y: {val_cat_y}")

                                        if len(train_cat_y) < 2:
                                            continue

                                        try:
                                            model = SARIMAX(train_cat_y, order=current_order, seasonal_order=current_seasonal,
                                                            enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                                            forecast = model.forecast(steps=len(val_cat_y))
                                            #print(f"forecast for category {cat}: {forecast}")
                                            rmse = root_mean_squared_error(val_cat_y, forecast)
                                            print(f"rmse for category {cat}: {rmse}")
                                            rmses.append(rmse)
                                        except:
                                            continue

                                    if rmses and np.mean(rmses) < best_rmse:
                                        best_rmse = np.mean(rmses)
                                        best_params = {"order": current_order, "seasonal_order": current_seasonal}

        print(f"Best SARIMA params: {best_params} with RMSE: {best_rmse}")
        return best_params, best_rmse
