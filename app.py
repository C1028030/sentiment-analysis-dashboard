import pandas as pd # Pandas is a Python library used for data manipulation and analysis. It provides data structures like DataFrames that make it easy to work with structured data.
import streamlit as st # Streamlit is a Python library that allows you to create interactive web applications for data science and machine learning projects. It provides an easy way to build user interfaces and visualize data.
import re # The re module in Python provides support for regular expressions, which are used for pattern matching and text manipulation. It allows you to search, match, and manipulate strings based on specific patterns.
from sklearn.feature_extraction.text import TfidfVectorizer # TfidfVectorizer is a class from the scikit-learn library that converts a collection of raw text documents into a matrix of TF-IDF features. It is commonly used in natural language processing tasks to represent text data numerically for machine learning models.
from sklearn.model_selection import train_test_split # train_test_split is a function from the scikit-learn library that splits a dataset into training and testing subsets. It is commonly used in machine learning to evaluate model performance by training on one subset and testing on another.
from sklearn.linear_model import LogisticRegression # Logistic Regression will learn to classify comments by sentiment
from sklearn.svm import LinearSVC # LinearSVC creates a Linear Support Vector Machine classifier
from sklearn.metrics import ( # These tools measure how well the classifier performs
    accuracy_score,
    classification_report,
    confusion_matrix
)
# These libraries will display the confusion matrix as a chart
import matplotlib.pyplot as plt
import seaborn as sns

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

# Create the logistic regression classification model
logistic_model = LogisticRegression(
    # Allow the traning process enough attempts to find a solution
    max_iter=1000,

    # Use a fixed value to make the model's rresults reproducible
    random_state=42
)

# Train the model using the TF-IDF features and correct sentiment labels
# During training, the model learns relationships between words and sentiments
logistic_model.fit(X_train_tfidf, y_train)

# Ask the trained model to predict the sentiments of the unseen test comments
logistic_predictions = logistic_model.predict(X_test_tfidf)

# Calculate the percentage of test comments classified correctly
logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)

# Create detailed performance measurements for each sentiment
logistic_classification_results = classification_report(
    y_test,
    logistic_predictions,

    # Return the results as a dictionary so Pandas can display them
    output_dict=True,

    # Prevent warnings if the model predicts none of a particular sentiment class
    zero_division=0
)

# Convert the classification results into a readable dataframe
logistic_results_df = pd.DataFrame(
    logistic_classification_results
).transpose()

# Set a consistent order for the sentiment classes
sentiment_labels = ["negative", "neutral", "positive"]

# Create a confusion matrix comparing the real and predicted labels
logistic_confusion_matrix = confusion_matrix(
    y_test,
    logistic_predictions,
    labels=sentiment_labels
)

# --- LINEAR SVM CLASSIFICATION ---

# Create the linear support vector machine model
linear_svm_model = LinearSVC(
    # Use a fixed value so teh results are reproducible
    random_state=42,

    # Allow enough iterations for the model to complete its training
    max_iter=5000
)

# Train Linear SVM using exactly the same training data used by Logistic Regression
linear_svm_model.fit(
    X_train_tfidf,
    y_train
)

# Predict the sentiment of the same unseen test comments
svm_predictions = linear_svm_model.predict(
    X_test_tfidf
)

# Calculate the percentage of correct SVM predictions
svm_accuracy = accuracy_score(
    y_test,
    svm_predictions
)

# Calculate precision, recall and F1-Score for Linear SVM
svm_classification_results = classification_report(
    y_test,
    svm_predictions,

    # Return the measurements in a dictionary
    output_dict=True,

    # Avoid warnings if a sentiment receives no predictions
    zero_division=0
)

# Convert the Linear SVM results into a readable dataframe
svm_results_df = pd.DataFrame(
    svm_classification_results
).transpose()

# Create a confusion matrix ffor Linear SVM
svm_confusion_matrix = confusion_matrix(
    y_test,
    svm_predictions,
    labels=sentiment_labels
)


# --- STREAMLIT USER INTERFACE ---

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

# Display the Logistic Regression evaluation section
st.subheader("Logistic Regression classification")

# Convert accuracy from a decimal into a percentage
st.metric(
    label="Model accuracy",
    value=f"{logistic_accuracy:.2%}"
)

st.write(
    "Accuracy represents the percentage of unseen test comments"
    "that the model classified correctly."
)

# Display precision, recall and F1-score for each sentiment class
st.subheader("Classification report")

st.dataframe(
    logistic_results_df.round(3),
    use_container_width=True
)

# Matplot figure for the confusion matrix
figure, axis = plt.subplots(figsize=(7, 5))

# Display the confusion matrix as a coloured heatmap
sns.heatmap(
    logistic_confusion_matrix,

    # Write the number from each cell onto the chart
    annot=True,

    # Display whole numbers instead of decimal values
    fmt="d",

    # Use a blue colour scheme
    cmap="Blues",

    # Add the sentiment names to both axes
    xticklabels=sentiment_labels,
    yticklabels=sentiment_labels,

    # Draw the chart on the axis created above
    ax=axis
)

# Explain what each axis represents
axis.set_xlabel("Predicted sentiment")
axis.set_ylabel("Actual sentiment")
axis.set_title("Logistic Regression confusion matrix")

# Display the completed chart in Streamlit
st.pyplot(figure)

# Close the figure after displaying it to avoid unnecessary memory use
plt.close(figure)

# Create a table containing test comments and the model's predictions
prediction_examples = pd.DataFrame({
    "Processed comment": X_test,
    "Actual sentiment": y_test,
    "Predicted sentiment": logistic_predictions
})

# Add a column showing whether each prediction was correct
prediction_examples["Correct prediction"] = (
    prediction_examples["Actual sentiment"]
    == prediction_examples["Predicted sentiment"]
)

st.subheader("Example model predictions")

st.dataframe(
    prediction_examples.head(20),
    use_container_width=True
)

# --- LINEAR SVM RESULTS ---

st.subheader("Linear SVM classification")

# Display the overall Linear SVM accuracy
st.metric(
    label="Linear SVM accuracy",
    value=f"{svm_accuracy:.2%}"
)

st.write(
    "Linear SVM was trained and tested using the same data as "
    "Logistic Regression, allowing the models to be compared fairly."
)

# Display precision, recall and F1-score
st.subheader("Linear SVM classification report")

st.dataframe(
    svm_results_df.round(3),
    use_container_width=True
)

# Create a figure for the Linear SVM confusion matrix
svm_figure, svm_axis = plt.subplots(figsize=(7, 5))

# Display the confusion matrix as a heatmap
sns.heatmap(
    svm_confusion_matrix,

    # Display the number of predictions in each cell
    annot=True,

    # Use whole numbers
    fmt="d",

    # Use a green colour scheme to distinguish it from the Logistic Regression Chart
    cmap="Greens",

    # Display the sentiment labels on both axes
    xticklabels=sentiment_labels,
    yticklabels=sentiment_labels,

    # Draw the heatmap on the axis created above
    ax=svm_axis
)

# Label the chart
svm_axis.set_xlabel("Predicted sentiment")
svm_axis.set_ylabel("Actual sentiment")
svm_axis.set_title("Linear SVM confusion matrix")

# Display the chart in Streamlit
st.pyplot(svm_figure)

# Close the figure after displaying it
plt.close(svm_figure)

# ---  MODEL COMPARISON ---
st.subheader("Classifier comparison")

# Extract the macro-average F1-score from each classification report
# Macro averaging gives every sentiment class equal importance
logistic_macro_f1 = logistic_classification_results[
    "macro avg"
]["f1-score"]

svm_macro_f1 = svm_classification_results[
    "macro avg"
]["f1-score"]

# Create a comparison table
model_comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Linear SVM"
    ],
    "Accuracy": [
        logistic_accuracy,
        svm_accuracy
    ],
    "Macro F1-score": [
        logistic_macro_f1,
        svm_macro_f1
    ]
})

# Display the scores as percentages
st.dataframe(
    model_comparison.style.format({
        "Accuracy": "{:.2%}",
        "Macro F1-score": "{:.2%}"
    }),
    use_container_width=True
)

# Identify which model achieved the higher accuracy
if svm_accuracy > logistic_accuracy:
    best_model = "Linear SVM"
elif logistic_accuracy > svm_accuracy:
    best_model = "Logistic Regression"
else:
    best_model = "Both models achieved the same accuracy"

st.write(
    f"**Highest accuracy:** {best_model}"
)