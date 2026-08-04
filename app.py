import pandas as pd # import numpy as np
import streamlit as st # import plotly.express as px

# Configure the web page
st.set_page_config(
    page_title="Youtube Comment Analysis",
    page_icon=":bar_chart:",
    layout="wide"
)

# Load the dataset
@st.cache_data # Cache the data to improve performance
def load_data():
    return pd.read_csv("data/youtube_comments_labels.csv")

dataset = load_data()

# Application heading
st.title("Youtube Comment Analysis") # title

st.write(
    "This application uses natural language processing and machine "
    "learning to analyse YouTube comments"
)

# Show basic dataset information
st.subheader("Dataset Overview") # subheader

column1, column2 = st.columns(2) 

with column1:
    st.metric("Number of comments", len(dataset))

with column2:
    st.metric("Number of columns", len(dataset.columns))

# Display the dataset
st.subheader("Comment data")
st.dataframe(dataset.head(20), use_container_width=True) # Display the first 20 rows of the dataset

# Show the missing values
st.subheader("Missing Values") # subheader
st.dataframe(
    dataset.isnull().sum().reset_index(
        name="Missing Values"
    ),
    use_container_width=True
)