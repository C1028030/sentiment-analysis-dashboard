# import pandas as pd # import numpy as np
# import streamlit as st # import plotly.express as px

# # Configure the web page
# st.set_page_config(
#     page_title="Youtube Comment Analysis",
#     page_icon=":bar_chart:",
#     layout="wide"
# )

# # Load the dataset
# @st.cache_data # Cache the data to improve performance
# def load_data():
#     return pd.read_csv("data/youtube_comments_labels.csv")

# dataset = load_data()

# # Application heading
# st.title("Youtube Comment Analysis") # title

# st.write(
#     "This application uses natural language processing and machine "
#     "learning to analyse YouTube comments"
# )

# # Show basic dataset information
# st.subheader("Dataset Overview") # subheader

# column1, column2 = st.columns(2) 

# with column1:
#     st.metric("Number of comments", len(dataset))

# with column2:
#     st.metric("Number of columns", len(dataset.columns))

# # Display the dataset
# st.subheader("Comment data")
# st.dataframe(dataset.head(20), use_container_width=True) # Display the first 20 rows of the dataset

# # Show the missing values
# st.subheader("Missing Values") # subheader
# st.dataframe(
#     dataset.isnull().sum().reset_index(
#         name="Missing Values"
#     ),
#     use_container_width=True
# )

import pandas as pd
import streamlit as st

# Configure the Streamlit browser page
st.set_page_config(
    page_title="YouTube Comment Analysis",
    page_icon="💬",
    layout="wide"
)


@st.cache_data
def load_and_clean_data():
    """
    Loads and cleans the YouTube comments dataset.

    A function is used so that the cleaning process is organised
    and can be reused elsewhere in the application.
    """

    comment_column = "Comments"
    label_column = "Sentiment"

    # Read the original CSV file into a Pandas DataFrame
    original_data = pd.read_csv("data/youtube_comments_labels.csv")

    # Create a copy so we do not accidentally change the original DataFrame
    cleaned_data = original_data.copy()

    # Remove unnecessary spaces from every column name
    # For example, " Comment " becomes "Comment"
    cleaned_data.columns = cleaned_data.columns.str.strip()

    # Remove rows where either the comment or human label is missing
    cleaned_data = cleaned_data.dropna(
        subset=[comment_column, label_column]
    )

    # Convert comments to strings and remove spaces from their ends
    cleaned_data[comment_column] = (
        cleaned_data[comment_column]
        .astype(str)
        .str.strip()
    )

    # Remove rows containing an empty comment
    cleaned_data = cleaned_data[
        cleaned_data[comment_column] != ""
    ]

    # Standardise the sentiment labels
    # For example, " Positive " and "POSITIVE" both become "positive"
    cleaned_data[label_column] = (
        cleaned_data[label_column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Keep only the three sentiment labels expected by the project
    valid_labels = ["positive", "neutral", "negative"]

    cleaned_data = cleaned_data[
        cleaned_data[label_column].isin(valid_labels)
    ]

    # Remove repeated comments
    # keep="first" retains the first occurrence and removes later copies
    cleaned_data = cleaned_data.drop_duplicates(
        subset=[comment_column],
        keep="first"
    )

    # Reset the row numbers after removing data
    cleaned_data = cleaned_data.reset_index(drop=True)

    return original_data, cleaned_data


# Run the function and store both versions of the dataset
original_data, cleaned_data = load_and_clean_data()


# Main page heading
st.title("YouTube Comment Analysis Dashboard")

st.write(
    "This application uses natural language processing and "
    "machine learning to analyse YouTube comments."
)


# Display a summary of the cleaning process
st.subheader("Data-cleaning summary")

column1, column2, column3 = st.columns(3)

with column1:
    st.metric(
        label="Original rows",
        value=len(original_data)
    )

with column2:
    st.metric(
        label="Cleaned rows",
        value=len(cleaned_data)
    )

with column3:
    # Calculate how many rows were removed during cleaning
    removed_rows = len(original_data) - len(cleaned_data)

    st.metric(
        label="Rows removed",
        value=removed_rows
    )


# Display the cleaned dataset
st.subheader("Cleaned dataset")

st.dataframe(
    cleaned_data.head(20),
    use_container_width=True
)


# Count how many comments belong to each sentiment class
st.subheader("Sentiment label distribution")

label_counts = (
    cleaned_data["Sentiment"]
    .value_counts()
    .rename_axis("Sentiment")
    .reset_index(name="Number of comments")
)

st.dataframe(
    label_counts,
    use_container_width=True
)