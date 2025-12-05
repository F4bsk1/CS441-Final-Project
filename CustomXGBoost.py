"""
XGBoost Predictor for Schnitzel Sales Forecasting

1. prepare_data() - Called ONCE to engineer features and store internally
2. find_best_params() - Uses stored data to tune hyperparameters
3. run_on_test() - Uses stored data to train and evaluate

All methods use the core fit(), predict(), evaluate() internally.
"""

from ModelPredictor import ModelPredictor
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder


# FEATURE ENGINEERING FUNCTIONS

def _detect_grouping(df):
    """Detect which grouping column is present."""
    for col in ["MAIN_GROUP", "PRODUCT_GROUP", "ARTICLE"]:
        if col in df.columns:
            return col
    return None


def _add_fourier_features(df, period, prefix):
    """Add sine/cosine features for seasonality."""
    t = df["DATE"].dt.dayofyear
    df[f"{prefix}_sin"] = np.sin(2 * np.pi * t / period)
    df[f"{prefix}_cos"] = np.cos(2 * np.pi * t / period)
    return df


def _engineer_features(df, grouping, pred_horizon):
    """
    Create time-series features for a given grouping and pred_horizon.
    Args:
        pred_horizon: Days ahead we're predicting (from GUI). Determines min lag.
    
    Features created (for pred_horizon=7):
    - Calendar: DAY_OF_WEEK, IS_WEEKEND, WEEK_OF_YEAR, MONTH, DAY_OF_YEAR
    - Fourier: weekly and yearly sine/cosine
    - Lags: lag_7, lag_14, lag_21 (minimum = pred_horizon)
    - Rolling: shifted by pred_horizon first, then 7/14 day windows
    - Ratios: volatility, lag ratios using dynamic lag columns
    """
    df = df.copy()
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values(["DATE", grouping])

    # Calendar (from DATE only - no leakage)
    df["DAY_OF_WEEK"] = df["DATE"].dt.dayofweek
    df["IS_WEEKEND"] = (df["DAY_OF_WEEK"] >= 5).astype(int)
    df["WEEK_OF_YEAR"] = df["DATE"].dt.isocalendar().week.astype(int)
    df["MONTH"] = df["DATE"].dt.month
    df["DAY_OF_YEAR"] = df["DATE"].dt.dayofyear

    # Fourier (from DATE only - no leakage)
    df = _add_fourier_features(df, period=7, prefix="weekly")
    df = _add_fourier_features(df, period=365, prefix="yearly") #this will only be relevant for the big dataset

    # Lags - minimum lag = pred_horizon to avoid leakage
    # thinking correctly?? pred_horizon=5 → lag_5, lag_12, lag_19
    grouped = df.groupby(grouping)
    lag_h = pred_horizon
    lag_h7 = pred_horizon + 7
    lag_h14 = pred_horizon + 14
    
    df[f"lag_{lag_h}"] = grouped["QUANTITY"].shift(lag_h)
    df[f"lag_{lag_h7}"] = grouped["QUANTITY"].shift(lag_h7)
    df[f"lag_{lag_h14}"] = grouped["QUANTITY"].shift(lag_h14)

    # Rolling - shift by pred_horizon FIRST to avoid leakage
    for window in [7, 14]:
        shifted = grouped["QUANTITY"].shift(pred_horizon)
        df[f"roll_mean_{window}"] = shifted.rolling(window).mean()
        df[f"roll_std_{window}"] = shifted.rolling(window).std()
        df[f"roll_max_{window}"] = shifted.rolling(window).max()
        df[f"roll_min_{window}"] = shifted.rolling(window).min()

    # Ratios (using dynamic lag column names)
    df["volatility_ratio"] = df["roll_std_7"] / (df["roll_mean_7"] + 1e-5)
    df["lag_ratio_h_h7"] = df[f"lag_{lag_h}"] / (df[f"lag_{lag_h7}"] + 1e-5)
    df["lag_ratio_h7_h14"] = df[f"lag_{lag_h7}"] / (df[f"lag_{lag_h14}"] + 1e-5)
    df["lag_delta_h_h7"] = df[f"lag_{lag_h}"] - df[f"lag_{lag_h7}"]
    df["lag_delta_h7_h14"] = df[f"lag_{lag_h7}"] - df[f"lag_{lag_h14}"]

    # Drop rows with NaN in features
    required = [f"lag_{lag_h}", f"lag_{lag_h7}", "roll_mean_7", "roll_std_7"]
    rows_before = len(df)
    df = df.dropna(subset=[c for c in required if c in df.columns])
    rows_after = len(df)
    print(f"[Feature Engineering] Dropped {rows_before - rows_after} rows ({100*(rows_before-rows_after)/rows_before:.1f}%) due to insufficient history")
    print(f"[Feature Engineering] Remaining rows: {rows_after}")
    
    # Check for remaining NaN and fill with 0
    nan_counts = df.isna().sum()
    if nan_counts.sum() > 0:
        print(f"[Feature Engineering] Filling {nan_counts.sum()} remaining NaN values with 0")
    df = df.fillna(0)
    
    return df


def _prepare_splits(train_df, val_df, test_df, grouping, pred_horizon):
    """
    Engineer features across all splits.
    
    Combines data so lags work across boundaries, then splits back.
    pred_horizon is passed through to ensure no data leakage.
    """
    combined = pd.concat([
        train_df.assign(SPLIT="TRAIN"),
        val_df.assign(SPLIT="VALIDATION"),
        test_df.assign(SPLIT="TEST"),
    ], ignore_index=True)
    
    engineered = _engineer_features(combined, grouping=grouping, pred_horizon=pred_horizon)
    
    train_ready = engineered[engineered["SPLIT"] == "TRAIN"].drop(columns=["SPLIT"])
    val_ready = engineered[engineered["SPLIT"] == "VALIDATION"].drop(columns=["SPLIT"])
    test_ready = engineered[engineered["SPLIT"] == "TEST"].drop(columns=["SPLIT"])
    
    return train_ready, val_ready, test_ready


def _encode_categoricals(train_df, val_df, test_df, cols):
    """Encode categoricals - fit on train only."""
    train_df, val_df, test_df = train_df.copy(), val_df.copy(), test_df.copy()
    
    for col in cols:
        if col not in train_df.columns:
            continue
        if train_df[col].nunique() == 0:
            train_df[col] = val_df[col] = test_df[col] = 0
            continue
        
        encoder = LabelEncoder().fit(train_df[col])
        train_df[col] = encoder.transform(train_df[col])
        
        fallback = encoder.classes_[0]
        for df in [val_df, test_df]:
            df[col] = encoder.transform(
                df[col].where(df[col].isin(encoder.classes_), fallback)
            )
    
    return train_df, val_df, test_df


# XGBOOST PREDICTOR CLASS

class XGBoostPredictor(ModelPredictor):
    """
    XGBoost predictor with state management.
    
    Usage:
        predictor = XGBoostPredictor()
        predictor.prepare_data(train, val, test)  # Called once
        best_params, rmse = predictor.find_best_params()
        results, metrics = predictor.run_on_test(best_params)
    """
    
    def __init__(self, pred_horizon=7):
        super().__init__("XGBoost")
        self.pred_horizon = pred_horizon
        
        # State - set by prepare_data()
        self._is_prepared = False
        self._grouping = None
        self._feature_cols = None
        
        # Engineered data splits
        self._train_eng = None
        self._val_eng = None
        self._test_eng = None
        
        # Encoded data splits (for training)
        self._train_enc = None
        self._val_enc = None
        self._test_enc = None
        
        # Model
        self._model = None

    # MAIN API
    
    def prepare_data(self, train_df, val_df, test_df):
        """
        Prepare and store engineered data. Call this ONCE.
        
        Uses self.pred_horizon (from GUI) to determine minimum lag for features.
        Returns the engineered DataFrames for display in Streamlit.
        """
        # Detect grouping
        self._grouping = _detect_grouping(train_df)
        
        # Engineer features with pred_horizon to avoid leakage
        self._train_eng, self._val_eng, self._test_eng = _prepare_splits(
            train_df, val_df, test_df, self._grouping, self.pred_horizon
        )
        
        # Determine feature columns
        self._feature_cols = [
            c for c in self._train_eng.columns 
            if c not in {"DATE", "QUANTITY"}
        ]
        
        # Encode categoricals (fit on train only)
        cat_cols = [self._grouping] if self._grouping in self._train_eng.columns else []
        self._train_enc, self._val_enc, self._test_enc = _encode_categoricals(
            self._train_eng, self._val_eng, self._test_eng, cat_cols
        )
        
        self._is_prepared = True
        
        # Return engineered data for Streamlit display
        return self._train_eng, self._val_eng, self._test_eng

    def find_best_params(self, progress_callback=None, extensive=True):
        """
        Find best hyperparameters using validation set.
        
        Args:
            progress_callback: Function to report progress
            extensive: If True, use larger grid (more combinations)
        
        Must call prepare_data() first.
        """
        if not self._is_prepared:
            raise RuntimeError("Call prepare_data() first")
        
        # Prepare X, y
        X_train = self._train_enc[self._feature_cols]
        y_train = self._train_enc["QUANTITY"]
        X_val = self._val_enc[self._feature_cols]
        y_val = self._val_enc["QUANTITY"]
        
        if extensive:
            # Extensive grid - ~72 combinations
            from itertools import product
            
            grid_values = {
                'n_estimators': [300, 500, 800],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'subsample': [0.7, 0.9],
                'colsample_bytree': [0.7, 0.9],
                'min_child_weight': [1, 3],
                'reg_lambda': [0.1, 1.0],
            }
            
            keys = list(grid_values.keys())
            param_grid = [dict(zip(keys, v)) for v in product(*grid_values.values())]
            print(f"Testing {len(param_grid)} hyperparameter combinations...")
        else:
            # Quick grid - 3 combinations
            param_grid = [
                {"n_estimators": 400, "learning_rate": 0.1, "max_depth": 4, "subsample": 0.8, "colsample_bytree": 0.8},
                {"n_estimators": 600, "learning_rate": 0.08, "max_depth": 6, "subsample": 0.85, "colsample_bytree": 0.85},
                {"n_estimators": 800, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.9},
            ]
        
        best_mae = float("inf")
        best_params = param_grid[0]
        
        for i, params in enumerate(param_grid):
            if progress_callback:
                progress_callback(i + 1, len(param_grid), f"Testing combination {i+1}/{len(param_grid)}")
            
            # Use core methods
            self.fit(X_train, y_train, params)
            preds = self.predict(X_val)
            _, _, mae = self.evaluate(y_val, preds)
            
            if mae < best_mae:
                best_mae = mae
                best_params = params.copy()
                print(f"New best (#{i+1}): MAE={mae:.4f}")
                print(f"  Params: {best_params}")
        
        print(f"\nBest params found: {best_params}")
        print(f"Best validation MAE: {best_mae:.4f}")
        
        return best_params, best_mae

    def run_on_test(self, best_params, progress_callback=None):
        """
        Train on train+val, evaluate on test.
        
        Must call prepare_data() first.
        """
        if not self._is_prepared:
            raise RuntimeError("Call prepare_data() first")
        
        # Combine train + val
        train_full = pd.concat([self._train_enc, self._val_enc], ignore_index=True)
        X_train = train_full[self._feature_cols]
        y_train = train_full["QUANTITY"]
        X_test = self._test_enc[self._feature_cols]
        y_test = self._test_enc["QUANTITY"]
        
        if progress_callback:
            progress_callback(1, 2, "Training XGBoost...")
        
        # Train using core fit()
        self.fit(X_train, y_train, best_params)
        
        # Predict using core predict()
        preds = self.predict(X_test)
        preds = np.maximum(preds, 0)
        preds = np.round(preds)
        
        # Evaluate using core evaluate()
        rmse, r2, mae = self.evaluate(y_test, preds)
        
        if progress_callback:
            progress_callback(2, 2, f"Done! MAE: {rmse:.3f}, R²: {r2:.3f}, MAE: {mae:.3f}")
        
        # Build results DataFrame
        results = self._test_eng.copy()
        results = results.rename(columns={"QUANTITY": "QUANTITY_TRUE"})
        results["QUANTITY_PREDICTIONS"] = preds
        
        return results, (rmse, r2, mae)

    # CORE METHODS - Used internally by find_best_params and run_on_test    
    def fit(self, X, y, params=None):
        """
        Train the XGBoost model.
        
        Args:
            X: Feature DataFrame or array
            y: Target Series or array
            params: Hyperparameters dict
        """
        params = params or {}
        
        self._model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=params.get("n_estimators", 400),
            max_depth=params.get("max_depth", 4),
            learning_rate=params.get("learning_rate", 0.1),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            reg_lambda=params.get("reg_lambda", 1.0),
            random_state=42,
        )
        self._model.fit(X, y)
        
        return self._model

    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Feature DataFrame or array
            
        Returns:
            numpy array of predictions
        """
        if self._model is None:
            raise RuntimeError("No model fitted - call fit() first")
        
        return self._model.predict(X)

    def evaluate(self, y_true, y_pred):
        """
        Calculate evaluation metrics.
        
        Returns:
            tuple: (RMSE, R², ME)
        """
        rmse = root_mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        return rmse, r2, mae

    def get_feature_importance(self):
        """
        Get feature importance from trained model.
        
        Returns:
            DataFrame with columns: feature, importance
            Sorted by importance descending.
        """
        if self._model is None:
            raise RuntimeError("No model trained yet - run training first")
        
        importance_df = pd.DataFrame({
            'feature': self._feature_cols,
            'importance': self._model.feature_importances_
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        
        return importance_df

    def drop_low_importance_features(self, threshold=0.005):
        """
        Drop features with importance below threshold and update internal state.
        
        Args:
            threshold: Minimum importance to keep (default 0.5% = 0.005)
            
        Returns:
            tuple: (features_dropped, features_kept)
        """
        if self._model is None:
            raise RuntimeError("No model trained yet - run training first")
        
        importance_df = self.get_feature_importance()
        
        # Identify features to drop
        low_importance = importance_df[importance_df['importance'] < threshold]
        high_importance = importance_df[importance_df['importance'] >= threshold]
        
        features_dropped = low_importance['feature'].tolist()
        features_kept = high_importance['feature'].tolist()
        
        if not features_dropped:
            return [], features_kept
        
        # Update feature columns
        self._feature_cols = features_kept
        
        # Re-encode with only kept features
        # (The encoded dataframes still have all columns, we just use fewer)
        
        print(f"Dropped {len(features_dropped)} features below {threshold*100:.1f}% importance:")
        for f in features_dropped:
            imp = importance_df[importance_df['feature'] == f]['importance'].values[0]
            print(f"  - {f}: {imp*100:.2f}%")
        
        return features_dropped, features_kept

    # BASE CLASS COMPATIBILITY - For Streamlit's generic interface
    # These wrap our main API to match the ModelPredictor interface    
    def find_best_params_compat(self, train_df, val_df, test_df, hyperparameter_list, progress_callback=None):
        """
        Compatibility wrapper for base class interface.
        Called by Streamlit via model_predictor.find_best_params()
        """
        # If not prepared, prepare now (shouldn't happen with new flow)
        if not self._is_prepared:
            self.prepare_data(train_df, val_df, test_df)
        
        return self.find_best_params(progress_callback)

    def run_on_test_compat(self, train_df, val_df, test_df, best_params, progress_callback=None):
        """
        Compatibility wrapper for base class interface.
        Called by Streamlit via model_predictor.run_on_test()
        """
        # If not prepared, prepare now (shouldn't happen with new flow)
        if not self._is_prepared:
            self.prepare_data(train_df, val_df, test_df)
        
        return self.run_on_test(best_params, progress_callback)

    # Override base class methods to use our implementation
    def tune_hyperparameters(self, train_X, train_y, val_X, val_y, hyperparameter_list, progress_callback=None):
        """Base class requirement - not used in our flow."""
        return {"n_estimators": 400, "learning_rate": 0.1, "max_depth": 4}, 0.0
