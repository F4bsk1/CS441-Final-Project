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

    cols = st.columns(4)
    with cols[0]:
        test_split_days = st.number_input("Number of days for test set:", key="test_split_days", value=7)
    with cols[1]:
        val_split_days = st.number_input("Number of days for validation set:", key="val_split_days", value=7)
    with cols[2]:
        grouping = st.selectbox("Select grouping for predictions:", options=['ARTICLE', 'PRODUCT_GROUP', 'MAIN_GROUP', 'NONE'], key="grouping")
    with cols[3]:
        model = st.selectbox("Select prediction model:", options=['XGBoost', 'SARIMA', 'LSTM', 'Transformer'], key="model")

    if st.button("Create Data Splits"):
        dataset.create_split_annotated_dataset(val_split_days=val_split_days, test_split_days=test_split_days)
        st.session_state.train_set, st.session_state.validation_set, st.session_state.test_set = dataset.get_dataset_splits(grouping)
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
                
            }
        
    elif model == "SARIMA":
        st.session_state.param_grid = {
                "p": [0, 1, 3, 7, 8],
                "d": [0, 1, 2],
                "q": [0, 1, 2],
                "P": [0, 1, 2],
                "D": [0, 1],
                "Q": [0, 1, 2],
                "s": [7] 
            }
        st.session_state.best_params = {
            'order': (3, 2, 2), 
            'seasonal_order': (2, 1, 2, 7)
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

    if model == 'XGBoost':
        model_predictor = XGBoostPredictor(pred_horizon=test_split_days)
    elif model == 'SARIMA':
        model_predictor = SARIMAPredictor(pred_horizon=test_split_days)
    elif model == 'LSTM':
        pass
        # Placeholder for LSTM training logic
    elif model == 'Transformer':
        pass
        # Placeholder for Transformer training logic

    st.subheader("Model Training")
    if st.button("Find Best Params"):
        if not st.session_state.split_done:
            st.error("Please create data splits before training the model.")
            # Placeholder for model training logic
        else:
            st.text(f"Finding Best Params for {model} model on Validation Set...")
            st.session_state.best_params, st.session_state.best_val_rmse = model_predictor.find_best_params(st.session_state.train_set, st.session_state.validation_set, st.session_state.test_set, st.session_state.param_grid)

            st.success("Best parameters found!")
            st.session_state.best_params_found = True

    if st.button("Prediction and Test Evaluation"):
        if not st.session_state.split_done:
            st.error("Please create data splits before training the model.")
            # Placeholder for model training logic
        else:
            st.text(f"Predicting Data...")
            print(st.session_state.best_params)
            st.session_state.results_test, st.session_state.eval = model_predictor.run_on_test(st.session_state.train_set, st.session_state.validation_set, st.session_state.test_set, st.session_state.best_params)
            st.success("Model trained successfully!")
            st.session_state.model_trained = True
            #st.session_state.results_pred = model_predictor.predict_future(st.session_state.train_set, st.session_state.validation_set, st.session_state.test_set)

    st.subheader("Model Evaluation")
    if st.session_state.best_params_found:
        st.text(f"**Evaluation Metrics - Validation Set:**")
        st.text(f"Best Hyperparameters: {st.session_state.best_params}")
        st.text(f"Best Validation RMSE during tuning: {st.session_state.best_val_rmse}")
    if st.session_state.model_trained:
        if st.toggle("Show Result Dataset - Test"):
            st.dataframe(st.session_state.results_test)
        st.text(f"**Evaluation Metrics - Test Set:**")
        st.text(f"Root Mean Squared Error (RMSE): {st.session_state.eval[0]}")
        st.text(f"R-squared (R2) Score: {st.session_state.eval[1]}")

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
        #st.dataframe(df_transformed)

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