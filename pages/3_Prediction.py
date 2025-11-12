import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from XGBoost import XGBoostPredictor
from ModelPredictor import ModelPredictor

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

    cols = st.columns(4)
    with cols[0]:
        test_split_days = st.number_input("Number of days for test set:", key="test_split_days", value=7)
    with cols[1]:
        val_split_days = st.number_input("Number of days for validation set:", key="val_split_days", value=7)
    with cols[2]:
        grouping = st.selectbox("Select grouping for predictions:", options=['ARTICLE', 'PRODUCT_GROUP', 'MAIN_GROUP', 'NONE'], key="grouping")
    with cols[3]:
        model = st.selectbox("Select prediction model:", options=['XGBoost', 'LSTM', 'Transformer'], key="model")

    if st.button("Create Data Splits"):
        dataset.create_split_annotated_dataset(val_split_days=val_split_days, test_split_days=test_split_days)
        st.success("Data splits created successfully!")
        st.session_state.split_done = True

    if st.toggle("Show Split Dataset"):
        try:
            train_set, validation_set, test_set = dataset.get_dataset_splits(grouping)
            
            st.subheader("Training Set")
            st.text(f"Training set contains {train_set.shape[0]} records from {train_set['DATE'].min().date()} to {train_set['DATE'].max().date()}.")
            st.dataframe(train_set)
            st.subheader("Validation Set")
            st.text(f"Validation set contains {validation_set.shape[0]} records from {validation_set['DATE'].min().date()} to {validation_set['DATE'].max().date()}.")
            st.dataframe(validation_set)
            st.subheader("Test Set")
            st.text(f"Test set contains {test_set.shape[0]} records from {test_set['DATE'].min().date()} to {test_set['DATE'].max().date()}.")
            st.dataframe(test_set)

        except ValueError as e:
            st.error(str(e))

    st.subheader("Model Training")
    if st.button("Train Model"):
        if not st.session_state.split_done:
            st.error("Please create data splits before training the model.")
            # Placeholder for model training logic
        else:
            if model == 'XGBoost':
                st.text("Training XGBoost model...")
                # Placeholder for XGBoost training logic
                model_predictor = XGBoostPredictor(hyperparameter_list={'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.2]})
                results = model_predictor.run(train_set, validation_set, test_set)
                st.text(results)
            elif model == 'LSTM':
                st.text("Training LSTM model...")
                # Placeholder for LSTM training logic
            elif model == 'Transformer':
                st.text("Training Transformer model...")
                # Placeholder for Transformer training logic
            st.success("Model trained successfully!")
            st.session_state.model_trained = True

    st.subheader("Model Evaluation")

    st.subheader("Show Predictions")

else:
    st.warning("Preprocessed data not found. Please run preprocessing first.")