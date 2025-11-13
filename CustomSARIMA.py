from ModelPredictor import ModelPredictor
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import root_mean_squared_error, r2_score

class SARIMAPredictor(ModelPredictor):
    def __init__(self, hyperparameter_list=None, pred_horizon=7):
        super().__init__("SARIMA", pred_horizon)
        self.models_per_category = {}
        self.pred_horizon = pred_horizon
        self.hyperparameter_list = hyperparameter_list or {}
    
    def _get_grouping_column(self, df):
        if 'MAIN_GROUP' in df.columns:
            return 'MAIN_GROUP'
        elif 'PRODUCT_GROUP' in df.columns:
            return 'PRODUCT_GROUP'
        elif 'ARTICLE' in df.columns:
            return 'ARTICLE'
        else:
            raise ValueError("No valid grouping column found")

    def fit(self, train_X, train_y, hyperparameters=None, store_model=True):
        train_X_copy = train_X.copy()
        train_y_copy = train_y.copy()
        grouping = self._get_grouping_column(train_X_copy)

        print(f"hyperparameters: {hyperparameters}")
        order = hyperparameters.get("order", (3, 0, 2))
        seasonal_order = hyperparameters.get("seasonal_order", (2,1,1,7))
        print(f"order: {order}, seasonal_order: {seasonal_order}")
        categories = train_X_copy[grouping].unique()

        for cat in categories:
            cat_mask = train_X_copy[grouping] == cat
            cat_data = pd.DataFrame({
                'DATE': train_X_copy.loc[cat_mask, 'DATE'].values,
                'QUANTITY': train_y_copy[cat_mask].values
            })
            cat_data = cat_data.sort_values('DATE').reset_index(drop=True)
            cat_y = cat_data['QUANTITY'].values
            
            try:
                model = SARIMAX(cat_y, order=order, seasonal_order=seasonal_order, 
                              enforce_stationarity=False, enforce_invertibility=False)
                fit_model = model.fit(disp=False, maxiter=200)
                if store_model:
                    self.models_per_category[cat] = fit_model
            except Exception as e:
                print(f"Error fitting {cat}: {e}")
                continue

        return self.models_per_category

    def predict(self, test_X):
        preds = np.zeros(len(test_X))
        test_X_copy = test_X.copy().reset_index(drop=True)
        grouping = self._get_grouping_column(test_X_copy)
        pred_df = pd.DataFrame()
        for cat, model in self.models_per_category.items():
            cat_mask = test_X_copy[grouping] == cat

            cat_test = pd.DataFrame({
                'DATE': test_X_copy.loc[cat_mask, 'DATE'].values,
                grouping: cat
            })
            cat_test = cat_test.sort_values('DATE').reset_index(drop=True)
            
            n_test_samples = len(cat_test)
            
            try:
                forecast = model.get_forecast(steps=n_test_samples).predicted_mean
                cat_test['QUANTITY'] = forecast
                pred_df = pd.concat([pred_df, cat_test])
            except Exception as e:
                print(f"Error predicting {cat}: {e}")
                continue

        return pred_df

    def evaluate(self, y_true, predictions):
        predictions = np.nan_to_num(predictions, nan=0.0)
        y_true = np.nan_to_num(y_true, nan=0.0)
        rmse = root_mean_squared_error(y_true, predictions)
        r2 = r2_score(y_true, predictions)
        return rmse, r2

    def tune_hyperparameters(self, train_X, train_y, val_X, val_y, hyperparameter_list):
        best_rmse = float("inf")
        best_params = {
            "order": (1, 1, 0),
            "seasonal_order": (0, 1, 1, 7)
        }

        grouping = self._get_grouping_column(train_X)
        p_values = hyperparameter_list.get("p", [0, 1])
        d_values = hyperparameter_list.get("d", [0, 1])
        q_values = hyperparameter_list.get("q", [0, 1])
        P_values = hyperparameter_list.get("P", [0, 1])
        D_values = hyperparameter_list.get("D", [0, 1])
        Q_values = hyperparameter_list.get("Q", [0, 1])
        s_values = hyperparameter_list.get("s", [7])

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
                                    print(f"Testing {current_order}, {current_seasonal}")

                                    for cat in categories:
                                        train_cat_mask = train_X[grouping] == cat
                                        val_cat_mask = val_X[grouping] == cat
                                        
                                        train_cat_data = pd.DataFrame({
                                            'DATE': train_X.loc[train_cat_mask, 'DATE'].values,
                                            'QUANTITY': train_y[train_cat_mask].values
                                        })
                                        train_cat_data = train_cat_data.sort_values('DATE').reset_index(drop=True)
                                        train_cat_y = train_cat_data['QUANTITY'].values
                                        
                                        val_cat_data = pd.DataFrame({
                                            'DATE': val_X.loc[val_cat_mask, 'DATE'].values,
                                            'QUANTITY': val_y[val_cat_mask].values
                                        })
                                        val_cat_data = val_cat_data.sort_values('DATE').reset_index(drop=True)
                                        val_cat_y = val_cat_data['QUANTITY'].values


                                        try:
                                            model = SARIMAX(train_cat_y, order=current_order, seasonal_order=current_seasonal,
                                                            enforce_stationarity=False, enforce_invertibility=False)
                                            fitted_model = model.fit(disp=False, maxiter=200)
                                            forecast = fitted_model.forecast(steps=len(val_cat_y))
                                            rmse = root_mean_squared_error(val_cat_y, forecast)
                                            rmses.append(rmse)
                                        except Exception as e:
                                            continue

                                    if rmses and np.mean(rmses) < best_rmse:
                                        best_rmse = np.mean(rmses)
                                        best_params = {"order": current_order, "seasonal_order": current_seasonal}
                                        print(f"new best: {best_params} rmse: {best_rmse}")

        print(f"Best params: {best_params} RMSE: {best_rmse}")
        return best_params, best_rmse
