import pandas as pd # Pandas is a Python library used for data manipulation and analysis. It provides data structures like DataFrames that make it easy to work with structured data.
import streamlit as st # Streamlit is a Python library that allows you to create interactive web applications for data science and machine learning projects. It provides an easy way to build user interfaces and visualize data.
import re # The re module in Python provides support for regular expressions, which are used for pattern matching and text manipulation. It allows you to search, match, and manipulate strings based on specific patterns.
from sklearn.feature_extraction.text import TfidfVectorizer # TfidfVectorizer is a class from the scikit-learn library that converts a collection of raw text documents into a matrix of TF-IDF features. It is commonly used in natural language processing tasks to represent text data numerically for machine learning models.
from sklearn.model_selection import train_test_split # train_test_split is a function from the scikit-learn library that splits a dataset into training and testing subsets. It is commonly used in machine learning to evaluate model performance by training on one subset and testing on another.

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

def preprocess_text(comment):
    """
    Prepares a comment for machine-learning analysis. The original comment is kept in the dataset so it can still be displayed to the user. This function creates a separate cleaned version specifically for TF-IDF and classification.
    """
    # Convert the comment to lowercase so words such as "Excellent" and "excellent" are treated as the same word
    comment = comment.lower()

    # Remove website links because URLs usually don't help the model understand the sentiment of a comment
    comment = re.sub(r"http\S+|www\.\S+", "", comment)

    # Remove the @ symbol from usernames so that "@username" becomes "username"
    comment = re.sub(r"@", "", comment)

    # Remove characters that are not letters, numbers, or spaces
    comment = re.sub(r"[^a-z0-9\s]", "", comment)

    # Replace repeated spaces with one space
    comment = re.sub(r"\s+", " ", comment)

    # Remove spaces from the beginning and end
    return comment.strip()

# Run the function and store both versions of the dataset
original_data, cleaned_data = load_and_clean_data()

# Apply the NLP preprocessing function to every comment
# The result is stored in a new column so the originalk text is preserved
cleaned_data["Processed Comments"] = cleaned_data["Comments"].apply(
    preprocess_text
)

# Remove any rows that became empty after NLP preprocessing
cleaned_data = cleaned_data[
    cleaned_data["Processed Comments"] != ""
].reset_index(drop=True)

# X contains the input text that the model will analyse
X = cleaned_data["Processed Comments"]

# y contains the correct sentiment labels that the model will learn to predict
y = cleaned_data["Sentiment"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,

    # Use 80% of the comments for training and 20% for testing
    test_size=0.20,

    # Keep the proportion of positive, neutral and negative comments similar in both sets
    stratify=y,

    # Use a fixed value so the split is reproducible
    random_state=42
)

# The TF-IDF vectoriser converts the text into a numerical representation that the machine learning model can understand. It calculates the importance of each word in a comment relative to the entire dataset.
tfidf_vectorizer = TfidfVectorizer(
    # Ignore common English words such as "the" and "and"
    stop_words="english",

    # Include individual words and pairs of consecutive words. For example: "good" and "very good"
    ngram_range=(1, 2),

    # Ignore words that appear in fewer that two comments
    min_df=2,

    # Limit the number of features to keep the baseline manageable
    max_features=5000
)

# Learn the vocabulary from the training comments and transform them
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Transform the test comments using the vocabulary learned from training data. We use transform(), not fit_transform(), to avoid  data leakage
X_test_tfidf = tfidf_vectorizer.transform(X_test)


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

# Display information about the NLP and TF-IDF stage
st.subheader("NLP and TF-IDF summary")

nlp_column1, nlp_column2, nlp_column3 = st.columns(3)

with nlp_column1:
    st.metric(
        label="Training comments",
        value=len(X_train)
    )

with nlp_column2:
    st.metric(
        label="Testing comments",
        value=len(X_test)
    )

with nlp_column3:
    st.metric(
        label="TF-IDF features",
        value=X_train_tfidf.shape[1]
    )


# Show examples of original and processed comments
st.subheader("NLP preprocessing examples")

st.dataframe(
    cleaned_data[
        ["Comments", "Processed Comments", "Sentiment"]
    ].head(10),
    use_container_width=True
)


# Retrieve the words and word pairs learned by TF-IDF
feature_names = tfidf_vectorizer.get_feature_names_out()

st.subheader("Example TF-IDF features")

# Display only the first 50 features to avoid filling the page
st.write(feature_names[:50])