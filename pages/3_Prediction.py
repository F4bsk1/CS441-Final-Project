import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from CustomXGBoost import XGBoostPredictor
from ModelPredictor import ModelPredictor
from CustomSARIMA import SARIMAPredictor
import json

st.set_page_config(
    page_title="Predictions",
    layout="wide",    # this makes the page use the full width
    initial_sidebar_state="expanded"  # optional
)

st.title("Predictions")
if "schnitzelPredictorDataset" in st.session_state:
    st.subheader("Select Parameters and create data splits")
    dataset = st.session_state.schnitzelPredictorDataset
    min_date, max_date = dataset.get_min_max_date()
    st.text(f"The first date in the dataset is {min_date.date()} and the last date is {max_date.date()}.")
    if "split_done" not in st.session_state:
        st.session_state.split_done = False
    if "model_trained" not in st.session_state:
        st.session_state.model_trained = False
    if "best_params_found" not in st.session_state:
        st.session_state.best_params_found = False

    cols = st.columns(5)
    with cols[0]:
        test_split_days = st.number_input("Number of days for test set:", key="test_split_days", value=7)
    with cols[1]:
        val_split_days = st.number_input("Number of days for validation set:", key="val_split_days", value=7)
    with cols[2]:
        pred_horizon = st.number_input("Predictor horizon:", value=7)
    with cols[3]:
        grouping = st.selectbox("Select grouping for predictions:", options=['ARTICLE', 'PRODUCT_GROUP', 'MAIN_GROUP', 'NONE'], key="grouping")
    with cols[4]:
        model = st.selectbox("Select prediction model:", options=['XGBoost', 'SARIMA'], key="model")

    if st.button("Create Data Splits"):
        dataset.create_split_annotated_dataset(val_split_days=val_split_days, test_split_days=test_split_days)
        
        # Get raw splits from dataset
        raw_train, raw_val, raw_test = dataset.get_dataset_splits(grouping)
        
        if model == "XGBoost":
            # XGBoost: Create predictor, prepare data ONCE, store in session_state
            xgb_predictor = XGBoostPredictor(pred_horizon=pred_horizon) #pred_horizon is the number of days to predict ahead
            
            # prepare_data() engineers features and stores them internally
            # Returns engineered data for display
            eng_train, eng_val, eng_test = xgb_predictor.prepare_data(raw_train, raw_val, raw_test)
            
            # Store predictor in session_state (so state persists across button clicks)
            st.session_state.xgb_predictor = xgb_predictor
            
            # Store engineered data for display
            st.session_state.train_set = eng_train
            st.session_state.validation_set = eng_val
            st.session_state.test_set = eng_test
        else:
            # SARIMA: Use raw data (unchanged from original)
            st.session_state.train_set = raw_train
            st.session_state.validation_set = raw_val
            st.session_state.test_set = raw_test
        
        st.success("Data splits created successfully!")
        st.session_state.split_done = True
        
    if model == "XGBoost": 
        st.session_state.param_grid = {
                            "n_estimators": [300, 600],
                            "learning_rate": [0.05, 0.1],
                            "max_depth": [3, 5],
                            "min_child_weight": [1, 3],
                            "subsample": [0.8],
                            "colsample_bytree": [0.8],
                            "reg_alpha": [0, 0.1],
                            "reg_lambda": [0, 1, 5],
                            "booster": ["gbtree"],
                            "tree_method": ["hist"],
                        }
        st.session_state.best_params = {
            'objective': 'reg:squarederror', 
            'max_depth': 5, 
            'learning_rate': 0.1, 
            'n_estimators': 600, 
            'subsample': 0.8, 
            'colsample_bytree': 0.8, 
            'min_child_weight': 3, 
            'gamma': 0, 
            'reg_alpha': 0, 
            'reg_lambda': 1, 
            'booster': 'gbtree', 
            'tree_method': 'hist'
            }
        
    elif model == "SARIMA":
        st.session_state.param_grid = {
                "p": [0, 1, 3, 7, 8],
                "d": [0, 1, 2],
                "q": [0, 1, 2],
                "P": [0, 1, 2],
                "D": [0, 1],
                "Q": [0, 1, 2],
                "s": [7, 30, 52, 365]
            }
        st.session_state.best_params = {
            'order': (3, 0, 2), 
            'seasonal_order': (2, 1, 1, 7)
        }
    else:
        raise ValueError("No Model selected!")
    if st.toggle("Hyperparameter Settings"):
        
        cols = st.columns(2)
        with cols[0]:
            st.session_state.param_grid = json.loads(st.text_area("Parameter Grid", value=json.dumps(st.session_state.param_grid)))
        with cols[0]:
            st.session_state.best_params = json.loads(st.text_area("Best Parameters", value=json.dumps(st.session_state.best_params)))

    if st.toggle("Show Split Dataset"):
        try:
            
            st.subheader("Training Set")
            st.text(f"Training set contains {st.session_state.train_set.shape[0]} records from {st.session_state.train_set['DATE'].min().date()} to {st.session_state.train_set['DATE'].max().date()}.")
            st.dataframe(st.session_state.train_set)
            st.subheader("Validation Set")
            st.text(f"Validation set contains {st.session_state.validation_set.shape[0]} records from {st.session_state.validation_set['DATE'].min().date()} to {st.session_state.validation_set['DATE'].max().date()}.")
            st.dataframe(st.session_state.validation_set)
            st.subheader("Test Set")
            st.text(f"Test set contains {st.session_state.test_set.shape[0]} records from {st.session_state.test_set['DATE'].min().date()} to {st.session_state.test_set['DATE'].max().date()}.")
            st.dataframe(st.session_state.test_set)

        except ValueError as e:
            st.error(str(e))

    st.subheader("Model Training")
    
    # XGBoost tuning options
    if model == "XGBoost":
        extensive_tuning = st.checkbox("Extensive tuning (72 combinations, slower)", value=False)
        st.caption("Quick: 3 combinations | Extensive: tests n_estimators, learning_rate, max_depth, subsample, colsample, min_child_weight, reg_lambda")
    
    if st.button("Find Best Params"):
        if not st.session_state.split_done:
            st.error("Please create data splits before training the model.")
        else:
            st.text(f"Finding Best Params for {model} model on Validation Set...")
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(current, total, message):
                progress = current / total if total > 0 else 0
                progress_bar.progress(progress)
                status_text.text(f"{message} ({current}/{total})")
        
            if model == "XGBoost":
                # XGBoost: Use stored predictor (data already prepared)
                xgb_predictor = st.session_state.xgb_predictor
                st.session_state.best_params, st.session_state.best_val_mae = xgb_predictor.find_best_params(
                    update_progress, extensive=extensive_tuning
                )
            else:
                # SARIMA: Use base class interface (unchanged)
                model_predictor = SARIMAPredictor(pred_horizon=test_split_days)
                st.session_state.best_params, st.session_state.best_val_mae = model_predictor.find_best_params(
                    st.session_state.train_set, st.session_state.validation_set, 
                    st.session_state.test_set, st.session_state.param_grid, update_progress
                )
            
            progress_bar.progress(1.0)
            status_text.text("Complete!")
            st.success("Best parameters found!")
            st.session_state.best_params_found = True

    if st.button("Prediction and Test Evaluation"):
        if not st.session_state.split_done:
            st.error("Please create data splits before training the model.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(current, total, message):
                progress = current / total if total > 0 else 0
                progress_bar.progress(progress)
                status_text.text(f"{message} ({current}/{total})")
            
            if model == "XGBoost":
                # XGBoost: Use stored predictor (data already prepared)
                xgb_predictor = st.session_state.xgb_predictor
                st.session_state.results_test, st.session_state.eval = xgb_predictor.run_on_test(
                    st.session_state.best_params, update_progress
                )
            else:
                # SARIMA: Use base class interface (unchanged)
                model_predictor = SARIMAPredictor(pred_horizon=test_split_days)
                st.session_state.results_test, st.session_state.eval = model_predictor.run_on_test(
                    st.session_state.train_set, st.session_state.validation_set, 
                    st.session_state.test_set, st.session_state.best_params,
                    progress_callback=update_progress
                )
            
            progress_bar.progress(1.0)
            status_text.text("Complete!")
            st.success("Model trained successfully!")
            st.session_state.model_trained = True

    st.subheader("Model Evaluation")
    
    # Validation Results
    if st.session_state.best_params_found:
        st.markdown("#### Validation Set Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Validation MAE", f"{st.session_state.best_val_mae:.4f}")
        with col2:
            with st.expander("Best Hyperparameters"):
                st.json(st.session_state.best_params)
    
    # Test Results
    if st.session_state.model_trained:
        st.markdown("#### Test Set Results")
        
        # Metrics in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("RMSE", f"{st.session_state.eval[0]:.4f}")
        with col2:
            st.metric("R² Score", f"{st.session_state.eval[1]:.4f}")
        with col3:
            mae = st.session_state.eval[2]
            st.metric("Mean Absolute Error", f"{mae:.4f}")
        
        # Expandable sections
        with st.expander("Show Test Dataset"):
            st.dataframe(st.session_state.results_test)
        
        # Feature Importance (XGBoost only)
        if model == "XGBoost" and "xgb_predictor" in st.session_state:
            try:
                importance_df = st.session_state.xgb_predictor.get_feature_importance()
                with st.expander("Feature Importance"):
                    st.bar_chart(importance_df.set_index('feature')['importance'])
                    st.dataframe(importance_df)
                    
                    # Show low importance features
                    threshold = 0.005  # 0.5%
                    low_imp = importance_df[importance_df['importance'] < threshold]
                    if len(low_imp) > 0:
                        st.markdown(f"**{len(low_imp)} features below {threshold*100:.1f}% importance:**")
                        st.text(", ".join(low_imp['feature'].tolist()))
                        
                        if st.button("Drop low-importance features and retrain"):
                            dropped, kept = st.session_state.xgb_predictor.drop_low_importance_features(threshold)
                            st.success(f"Dropped {len(dropped)} features. Retraining with {len(kept)} features...")
                            
                            # Retrain
                            results, metrics = st.session_state.xgb_predictor.run_on_test(st.session_state.best_params)
                            st.session_state.results_test = results
                            st.session_state.eval = metrics
                            
                            st.success(f"Retrained! New RMSE: {metrics[0]:.4f}, R²: {metrics[1]:.4f}")
                            st.rerun()
                    else:
                        st.info("All features have importance >= 0.5%")
            except RuntimeError:
                pass  # Model not trained yet, skip feature importance

    st.subheader("Show Predictions")
    if st.session_state.model_trained:
        df_transformed = st.session_state.results_test.melt(
            id_vars=['DATE', grouping],
            value_vars=['QUANTITY_TRUE', 'QUANTITY_PREDICTIONS'],
            var_name='TYPE',
            value_name='QUANTITY'
        )
        df_val = st.session_state.validation_set.copy()
        df_val['TYPE'] = 'QUANTITY_TRUE'
        df_train = st.session_state.train_set.copy()
        df_train['TYPE'] = 'QUANTITY_TRUE'
        df_combined = pd.concat([df_train, df_val, df_transformed], ignore_index=False)

        if st.toggle("Show Dataframes for Prediction:"):
            st.text("Transformed Results DataFrame:")
            st.dataframe(df_transformed)
            st.text("Combined DataFrame for Visualization:")
            st.dataframe(df_combined)

        fig = px.line(
        df_combined,
        x='DATE',
        y='QUANTITY',
        color=grouping,  
        line_dash='TYPE',  
        title='Quantity Over Time by Product Group',
        markers=True
        )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Quantity",
            legend_title="Product Group / Type",
            template="seaborn",
            width=1200,
            height=700
        )

        st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Preprocessed data not found. Please run preprocessing first.")
