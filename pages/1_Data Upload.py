import streamlit as st
import pandas as pd
from preprocessing import Preprocessor


st.set_page_config(
    page_title="Upload Data",
    layout="wide",    # this makes the page use the full width
    initial_sidebar_state="expanded"  # optional
)

# Create a section for uploading files
st.header("Step 1: Upload CSV Files")

# Allow users to upload two separate CSV files
transaction_details = st.file_uploader("Upload Transaction Details File", type=["csv"], key="file_transaction_details")
transaction_header = st.file_uploader("Upload Transaction Header File", type=["csv"], key="file_transaction_header")
product_groups = st.file_uploader("Upload Product Groups File", type=["csv"], key="file_product_groups")
articles = st.file_uploader("Upload Articles File", type=["csv"], key="file_articles")

# Read and display the data if both files are uploaded
if transaction_details is not None and transaction_header is not None and product_groups is not None and articles is not None:
    # Read files into pandas DataFrames
    df_transaction_details = pd.read_csv(transaction_details, delimiter=';')
    df_transaction_header = pd.read_csv(transaction_header, delimiter=';')
    df_product_groups = pd.read_csv(product_groups, delimiter=';')
    df_articles = pd.read_csv(articles, delimiter=';')

    st.success("Files uploaded successfully!")

    # Show preview of the data
    st.subheader("Preview of Transaction Header")
    st.dataframe(df_transaction_header.head())

    st.subheader("Preview of Transaction Details")
    st.dataframe(df_transaction_details.head())

    st.subheader("Preview of Product Groups")
    st.dataframe(df_product_groups.head())

    st.subheader("Preview of Articles")
    st.dataframe(df_articles.head())

else:
    st.info("Please upload CSV files to continue.")

st.title("Data Analysis")

if st.button("Preprocess Data"):
    if transaction_details is not None and transaction_header is not None and product_groups is not None and articles is not None:
        preprocessor = Preprocessor(df_transaction_details, df_transaction_header, df_product_groups, df_articles)
        st.session_state.df_preprocessed = preprocessor.preprocess()
        
        #st.success("Data preprocessed successfully!")
        st.subheader("Preview of Preprocessed Data")
        st.dataframe(st.session_state.df_preprocessed.head())

        st.session_state.df_grouped_by_day_and_article = preprocessor.create_grouped_by_day_and_article(st.session_state.df_preprocessed)
        st.subheader("Preview of Grouped by Day and Article Data")
        st.dataframe(st.session_state.df_grouped_by_day_and_article.head())


        st.session_state.df_grouped_by_day_and_product_group = preprocessor.create_grouped_by_day_and_product_group(st.session_state.df_preprocessed)
        st.subheader("Preview of Grouped by Day and Product Group Data")
        st.dataframe(st.session_state.df_grouped_by_day_and_product_group.head())

    else:
        st.error("Please upload all required files before preprocessing.")