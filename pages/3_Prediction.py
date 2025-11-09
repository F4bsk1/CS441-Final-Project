import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(
    page_title="Predictions",
    layout="wide",    # this makes the page use the full width
    initial_sidebar_state="expanded"  # optional
)

st.title("Predictions")