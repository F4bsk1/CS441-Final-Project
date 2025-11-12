import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from SchnitzelPredictorDataset import SchnitzelPredictorDataset

st.set_page_config(
    page_title="Data Visualization",
    layout="wide",    # this makes the page use the full width
    initial_sidebar_state="expanded"  # optional
)

st.title("Visualize Data")


# Check if preprocessed data exists
if "schnitzelPredictorDataset" in st.session_state:
    st.subheader("Explore Preprocessed Data")

    df_preprocessed = st.session_state.schnitzelPredictorDataset.get_dataset()
    df_grouped_by_day_and_article = st.session_state.schnitzelPredictorDataset.get_grouped_dataset(grouping="ARTICLE")
    df_grouped_by_day_and_product_group = st.session_state.schnitzelPredictorDataset.get_grouped_dataset(grouping="PRODUCT_GROUP")
    df_grouped_by_day_and_main_group = st.session_state.schnitzelPredictorDataset.get_grouped_dataset(grouping="MAIN_GROUP")

    if st.toggle("Show Preprocessed Data"):
        st.subheader("Preview of Preprocessed Data")
        st.dataframe(df_preprocessed.head())
    if st.toggle("Show Grouped by Day and Article Data"):
        st.subheader("Grouped by Day and Article Data")
        st.dataframe(df_grouped_by_day_and_article.head())
    if st.toggle("Show Grouped by Day and Product Group Data"):
        st.subheader("Grouped by Day and Product Group Data")
        st.dataframe(df_grouped_by_day_and_product_group.head())    
    if st.toggle("Show Grouped by Day and Main Group Data"):
        st.subheader("Grouped by Day and Main Group Data")
        st.dataframe(df_grouped_by_day_and_main_group.head())

        # Create the line plot
    st.subheader("Explore Graph")

    if st.toggle("Show Quantity Over Time by Product Group"):
        st.subheader("Quantity Over Time by Product Group")
        fig = px.line(
            df_grouped_by_day_and_product_group,
            x='DATE',
            y='QUANTITY',
            color='PRODUCT_GROUP',
            title='Quantity Over Time by Product Group',
            markers=True
        )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Quantity",
            legend_title="Product Group",
            template="seaborn",
            width=1200,   # increase width
            height=700    # increase height
        )

        st.plotly_chart(fig, use_container_width=True)

    if st.toggle("Show Quantity Over Time by Main Group"):
        st.subheader("Quantity Over Time by Main Group")
        fig = px.line(
            df_grouped_by_day_and_main_group,
            x='DATE',
            y='QUANTITY',
            color='MAIN_GROUP',
            title='Quantity Over Time by Main Group',
            markers=True
        )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Quantity",
            legend_title="Product Group",
            template="seaborn",
            width=1200,   # increase width
            height=700    # increase height
        )

        st.plotly_chart(fig, use_container_width=True)



    # Example 1: Correlation heatmap
    #if st.checkbox("Show Correlation Heatmap"):
    #    st.subheader("Correlation Heatmap")
    #    fig, ax = plt.subplots(figsize=(8,6))
    #    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    #    st.pyplot(fig)
#
    # Example 2: Histogram for numeric columns
    #numeric_cols = df.select_dtypes(include="number").columns.tolist()
    #if numeric_cols:
    #    column_to_plot = st.selectbox("Select a numeric column for histogram", numeric_cols)
     #   fig, ax = plt.subplots()
     #   ax.hist(df[column_to_plot], bins=20, color="skyblue", edgecolor="black")
     #   ax.set_title(f"Histogram of {column_to_plot}")
     #   st.pyplot(fig)
    #else:
    #    st.info("No numeric columns found for histogram.")
#
    # Example 3: Scatter plot between two numeric columns
#    if len(numeric_cols) >= 2:
 #       x_col = st.selectbox("Select X-axis column", numeric_cols, index=0)
  #      y_col = st.selectbox("Select Y-axis column", numeric_cols, index=1)
   #     fig, ax = plt.subplots()
    #    ax.scatter(df[x_col], df[y_col], alpha=0.7)
     #   ax.set_xlabel(x_col)
     #   ax.set_ylabel(y_col)
     #   ax.set_title(f"{y_col} vs {x_col}")
     #   st.pyplot(fig)
else:
    st.warning("Preprocessed data not found. Please run preprocessing first.")
