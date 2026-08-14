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
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # VADER provides rule-based sentiment analysis without model training
from sklearn.cluster import KMeans # KMeans groups comments according to similarities in their text
from sklearn.metrics import silhouette_score # The silhouette score measures how clearly separateed the clusters are
from sklearn.decomposition import TruncatedSVD # TruncatedSVD reduces TF-IDF data to two dimensions for visualisation

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

    # Define the columns required by the application
    required_columns = {
        comment_column,
        label_column
    }

    # Identify any required columns that are missing
    missing_columns = required_columns.difference(
        cleaned_data.columns
    )

    # Stop the cleaning process with a clear explanation if the dataset does not contain the required columns
    if missing_columns:
        raise ValueError(
            "The dataset is missing these required columns: "
            + ", ".join(sorted(missing_columns))
        )

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

def convert_vader_score_to_label(compound_score):
    """
    Converts VADER's numerical compound score into one of the three sentiment labels used in the dataset
    """

    # Scores of 0.05 or higher are treated as positive
    if compound_score >= 0.05:
        return "positive"

    # Scores of -0.05 or lower are treated as negative
    elif compound_score <= -0.05:
        return "negative"

    # Scores between -0.05 and 0.05 are treated as neutral
    else:
        return "neutral"

# Attempt to load and validate the dataset
try:
    original_data, cleaned_data = load_and_clean_data()

# Display a clear error instead of a large traceback
except FileNotFoundError:
    st.error(
        "The dataset could not be found. Make sure "
        "'data/youtube_comments_labels.csv' exists."
    )
    st.stop()

except ValueError as error:
    st.error(
        f"Dataset validation error: {error}"
    )
    st.stop()

except pd.errors.EmptyDataError:
    st.error(
        "The dataset exists but does not contain any data."
    )
    st.stop()

# Stop the application if no usable comments remain after cleaning
if cleaned_data.empty:
    st.error(
        "No usable comments remain after dataset cleaning."
    )
    st.stop()

# Each sentiment must contain at least two comments because stratified train/test splitting needs multiple examples per class
sentiment_counts = cleaned_data[
    "Sentiment"
].value_counts()

if sentiment_counts.min() < 2:
    st.error(
        "Each sentiment category must contain at leasty two comments "
        "for stratified training and testing."
    )
    st.stop()

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


# -- VADER SENTIMENT ANALYSIS ---

# Create the VADER sentiment analyser
vader_analyzer = SentimentIntensityAnalyzer()

# Retrieve the original versions of the comments in the test set
# We use the indexes from X_test so VADER is evaluated on exactly the same comments as Logistic Regression and Linear SVM
vader_test_comments = cleaned_data.loc[
    X_test.index,
    "Comments"
]

vader_scores = vader_test_comments.apply(
    vader_analyzer.polarity_scores
)

# Extract the compound score from each VADER result
# The compound score summarises the overall sentiment from -1 to 1
vader_compound_scores = vader_scores.apply(
    lambda score: score["compound"]
)

# Convert each compound score into positive, neutral or negative
vader_predictions = vader_compound_scores.apply(
    convert_vader_score_to_label
)

# Calculate how many VADER predictions match the human labels
vader_accuracy = accuracy_score(
    y_test,
    vader_predictions
)

# Calculate precision, recall and F1-score for each sentiment
vader_classification_results = classification_report(
    y_test,
    vader_predictions,
    output_dict=True,
    zero_division=0
)

# Convert the results into a DataFrame for the dashboard
vader_results_df = pd.DataFrame(
    vader_classification_results
).transpose()

# Create VADER's confusion matrix
vader_confusion_matrix = confusion_matrix(
    y_test,
    vader_predictions,
    labels=sentiment_labels
)

# --- K-MEANS CLUSTERING ---

# Create a separate TF-IDF vectoriser for clustering
# This is fitted on all cleaned comments because clustering does not use human sentiment labels or require a train/test split
clustering_vectorizer = TfidfVectorizer(
    # Remove common English words
    stop_words="english",

    # Include individual words and two-word phrases
    ngram_range=(1, 2),

    # Ignore terms appearing in fewer than two comments
    min_df=2,

    # Limit the number of features
    max_features=5000
)

# Convert every processed comment into TF-IDF features
clustering_tfidf = clustering_vectorizer.fit_transform(
    cleaned_data["Processed Comments"]
)

# Test several possible numbers of clusters
# Start at 2 because silhouette score cannot be calculated when every comment belongs to one cluster
possible_cluster_numbers = range(2, 7)

# Store the result produced by each cluster number
clustering_evaluation = []

for number_of_clusters in possible_cluster_numbers:
    # Create a temporary K-means model
    temporary_kmeans = KMeans(
        n_clusters=number_of_clusters,

        # Use several initial starting positions and retain the best result
        n_init=10,

        # Make the results reproducible
        random_state=42
    )

    # Assign every comment to a temporary cluster
    temporary_cluster_labels = temporary_kmeans.fit_predict(
        clustering_tfidf
    )

    # Inertia measures the distance between commments and their assigned cluster centres. Lower values indiciate tighter clusters.
    inertia = temporary_kmeans.inertia_

    # Silhouette score measures cluster separation
    # Values closer to 1 generally indicate clearer clusters
    silhouette = silhouette_score(
        clustering_tfidf,
        temporary_cluster_labels
    )

    # Store the evaluation results
    clustering_evaluation.append({
        "Number of clusters": number_of_clusters,
        "Inertia": inertia,
        "Silhouette score": silhouette
    })

# Convert the evaluation results into a DataFrame
clustering_evaluation_df = pd.DataFrame(
    clustering_evaluation
)

# Select the cluster number with the highest silhouette score
best_cluster_row = clustering_evaluation_df[
    "Silhouette score"
].idxmax()

best_cluster_number = int(
    clustering_evaluation_df.loc[
        best_cluster_row,
        "Number of clusters"
    ]
)

# Create the final K-means model using the selected cluster number
kmeans_model = KMeans(
    n_clusters=best_cluster_number,
    n_init=10,
    random_state=42
)

# Fit the model and assign a cluster number to every comment
cleaned_data["Cluster"] = kmeans_model.fit_predict(
    clustering_tfidf
)

# Retrieve all words and phrases used by the clustering vectoriser
clustering_feature_names = (
    clustering_vectorizer.get_feature_names_out()
)

# Store the most important terms from each cluster
cluster_term_results = []

for cluster_number in range(best_cluster_number):
    # Retrieve the cluster centre
    cluster_centre = kmeans_model.cluster_centers_[
        cluster_number
    ]

    # Find the indexes of the ten terms with the highest values
    top_term_indexes = cluster_centre.argsort()[-10:][::-1]

    # Convert the indexes back into readable words and phrases
    top_terms = clustering_feature_names[
        top_term_indexes
    ]

    # Store the cluster and its most important terms
    cluster_term_results.append({
        "Cluster": cluster_number,
        "Top terms": ", ".join(top_terms)
    })

# Convert the cluster terms into a DataFrame
cluster_terms_df = pd.DataFrame(
    cluster_term_results
)

# Count how many comments were assigned to each cluster
cluster_counts = (
    cleaned_data["Cluster"]
    .value_counts()
    .sort_index()
    .rename_axis("Cluster")
    .reset_index(name="Number of comments")
)

# Compare cluster membership with the human sentiment labels
# Human labels are used only for interpretation after clustering
cluster_sentiment_table = pd.crosstab(
    cleaned_data["Cluster"],
    cleaned_data["Sentiment"]
)

cluster_visualiser = TruncatedSVD(
    n_components=2,
    random_state=42
)

# Create two-dimensional coordinates for every comment
cluster_coordinates = cluster_visualiser.fit_transform(
    clustering_tfidf
)

# Store the coordinates and cluster numbers in a DataFrame
cluster_plot_data = pd.DataFrame({
    "Dimension 1": cluster_coordinates[:, 0],
    "Dimension 2": cluster_coordinates[:, 1],
    "Cluster": cleaned_data["Cluster"].astype(str)
})

# --- STREAMLIT USER INTERFACE ---

# Main page heading
st.title("YouTube Comment Analysis Dashboard")

st.write(
    "This application uses natural language processing and "
    "machine learning to analyse YouTube comments."
)

# Tabs to organise the dashboard into separate sections
live_tab, data_tab, classifier_tab, vader_tab, clustering_tab = st.tabs([
    "Live Analysis",
    "Data and NLP",
    "Classifiers",
    "VADER",
    "Clustering"
])

with live_tab:

    # --- LIVE COMMENT ANALYSIS ---

    st.subheader("Analyse a new comment")

    st.write(
        "Enter a YouTube comment to compare predictions from Logistic "
        "Regression, Linear SVM, VADER and K-means clustering."
    )

    # Text box where the user can enter a comment
    user_comment = st.text_area(
        label="YouTube comment",
        placeholder="For example: I really enjoyed this video!",
        height=120,

        # Prevent excessively long input from affecting performance
        max_chars=5000
    )

    # Button that starts the analysis
    analyse_button = st.button(
        "Analyse comment",
        type="primary"
    )

    # Run this section only when the button is pressed
    if analyse_button:
        # Remove spaces from the beginning and end of the comment
        user_comment = user_comment.strip()

        # Prevent the user from submitting an empty comment
        if user_comment == "":
            st.warning(
                "Please enter a comment before selecting Analyse comment."
            )

        else:
            # Apply the same preprocessing used on the training comments
            processed_user_comment = preprocess_text(
                user_comment
            )

            # Preprocessing could remove everything if the input only contains punctuation, symbols or emojis
            if processed_user_comment == "":
                st.warning(
                    "The comment does not contain enough readable text "
                    "for the machine-learning models to analyse."
                )

            else:
                # Convert the processed comment into TF-IDF features transform() is used because the vectoriser has already learned
                # its vocabulary from the training data
                user_comment_tfidf = tfidf_vectorizer.transform(
                    [processed_user_comment]
                )

                # nnz means "number of non-zero values"
                # A value of zer omeans none of the comment's terms appeared in the vocabulary learned from the training dataset
                if user_comment_tfidf.nnz == 0:
                    st.warning(
                        "None of the words in this comment appear in the model's "
                        "training vocabulary. Logistic Regression and Linear SVM "
                        "predictions may therefore be unreliable."
                    )

                # Predict sentiment using Logistic Regression
                logistic_user_prediction = logistic_model.predict(
                    user_comment_tfidf
                )[0]

                # Predict sentiment using Linear SVM
                svm_user_prediction = linear_svm_model.predict(
                    user_comment_tfidf
                )[0]

                # VADER analyses the original comment because punctuation, capitalisation and emojis may affect its result
                vader_user_scores = vader_analyzer.polarity_scores(
                    user_comment
                )

                # Retrieve VADER's overall score from -1 to +1
                vader_user_compound = vader_user_scores[
                    "compound"
                ]

                # Convert the numerical VADER score into a sentiment label
                vader_user_prediction = convert_vader_score_to_label(
                    vader_user_compound
                )

                # Convert the comment using the clustering vectoriser
                user_comment_clustering_tfidf = (
                    clustering_vectorizer.transform(
                        [processed_user_comment]
                    )
                )

                # Assign the new comment to its closest K-means cluster
                user_cluster_prediction = kmeans_model.predict(
                    user_comment_clustering_tfidf
                )[0]

                # --- DISPLAY LIVE RESULTS ---

                st.subheader("Analysis results")

                # Create four areas for the prediction results
                result_column1, result_column2 = st.columns(2)
                result_column3, result_column4 = st.columns(2)

                with result_column1:
                    st.metric(
                        label="Logistic Regression",
                        value=logistic_user_prediction.title()
                    )

                with result_column2:
                    st.metric(
                        label="Linear SVM",
                        value=svm_user_prediction.title()
                    )

                with result_column3:
                    st.metric(
                        label="VADER",
                        value=vader_user_prediction.title()
                    )

                with result_column4:
                    st.metric(
                        label="K-means cluster",
                        value=f"Cluster {user_cluster_prediction}"
                    )

                # Show VADER's numerical compound score
                st.write(
                    f"**VADER compound score:** "
                    f"{vader_user_compound:.3f}"
                )

                # Find the most important terms for the assigned cluster
                user_cluster_terms = cluster_terms_df.loc[
                    cluster_terms_df["Cluster"]
                    == user_cluster_prediction,
                    "Top terms"
                ].iloc[0]

                st.write(
                    f"**Important terms in Cluster "
                    f"{user_cluster_prediction}:** "
                    f"{user_cluster_terms}"
                )

                # Store the three sentiment predictions in a list
                sentiment_predictions = [
                    logistic_user_prediction,
                    svm_user_prediction,
                    vader_user_prediction
                ]

                # A set removes duplicate values. If its length is one, all three methods produced the same sentiment
                if len(set(sentiment_predictions)) == 1:
                    st.success(
                        "All three sentiment methods agree that this "
                        f"comment is {logistic_user_prediction}."
                    )

                else:
                    st.info(
                        "The sentiment methods produced different results. "
                        "This can happen because the machine-learning models "
                        "learn from the dataset, while VADER follows "
                        "predefined language rules."
                    )

with data_tab:

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

with classifier_tab:

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
    st.subheader("Sentiment method comparison")


    # Extract the macro F1-score for each method
    # Macro F1 gives equal importance to all three sentiment classes
    logistic_macro_f1 = logistic_classification_results[
        "macro avg"
    ]["f1-score"]

    svm_macro_f1 = svm_classification_results[
        "macro avg"
    ]["f1-score"]

    vader_macro_f1 = vader_classification_results[
        "macro avg"
    ]["f1-score"]


    # Create a table comparing all three sentiment methods
    model_comparison = pd.DataFrame({
        "Method": [
            "Logistic Regression",
            "Linear SVM",
            "VADER"
        ],
        "Approach": [
            "Supervised machine learning",
            "Supervised machine learning",
            "Rule-based sentiment analysis"
        ],
        "Accuracy": [
            logistic_accuracy,
            svm_accuracy,
            vader_accuracy
        ],
        "Macro F1-score": [
            logistic_macro_f1,
            svm_macro_f1,
            vader_macro_f1
        ]
    })


    # Sort the methods from highest to lowest macro F1-score
    model_comparison = model_comparison.sort_values(
        by="Macro F1-score",
        ascending=False
    ).reset_index(drop=True)


    # Display accuracy and macro F1-score as percentages
    st.dataframe(
        model_comparison.style.format({
            "Accuracy": "{:.2%}",
            "Macro F1-score": "{:.2%}"
        }),
        use_container_width=True
    )


    # Identify the method with the highest macro F1-score
    best_method = model_comparison.loc[0, "Method"]

    st.write(
        f"**Highest macro F1-score:** {best_method}"
    )

with vader_tab:

    # --- VADER RESULTS ---

    st.subheader("VADER sentiment analysis")

    # Display VADER's overall accuracy
    st.metric(
        label="VADER accuracy",
        value=f"{vader_accuracy:.2%}"
    )

    st.write(
        "VADER is a rule-based sentiment analyser. Unlike Logistic Regression and Linear SVM, it does not learn from this dataset."
    )

    # Display VADER's precision, recall and F1-score
    st.subheader("VADER classification report")

    st.dataframe(
        vader_results_df.round(3),
        use_container_width=True
    )

    # Create a figure for the VADER confusion matrix
    vader_figure, vader_axis = plt.subplots(
        figsize=(7, 5)
    )

    # Display the confusion matrix as a heatmap
    sns.heatmap(
        vader_confusion_matrix,

        # Display the number of predictions in each cell
        annot=True,

        # Display whole numbers
        fmt="d",

        # Use a different colour for VADER
        cmap="Oranges",

        # Use the same label order as the mother models
        xticklabels=sentiment_labels,
        yticklabels=sentiment_labels,

        # Draw the heatmap on this axis
        ax=vader_axis
    )

    # Label the chart
    vader_axis.set_xlabel("Predicted sentiment")
    vader_axis.set_ylabel("Actual sentiment")
    vader_axis.set_title("VADER confusion matrix")

    # Display and then close the figure
    st.pyplot(vader_figure)
    plt.close(vader_figure)

    # Table showing how VADER analysed individual comments
    vader_examples = pd.DataFrame({
        "Original comment": vader_test_comments,
        "Compound score": vader_compound_scores,
        "Actual sentiment": y_test,
        "VADER prediction": vader_predictions
    })

    # Show whether VADER agreed with the human label
    vader_examples["Correct prediction"] = (
        vader_examples["Actual sentiment"]
        == vader_examples["VADER prediction"]
    )

    st.subheader("Example VADER predictions")

    st.dataframe(
        vader_examples.head(20),
        use_container_width=True
    )

with clustering_tab:

    # --- K-MEANS CLUSTERING RESULTS ---

    st.subheader("K-means comment clustering")

    st.write(
        "K-means groups comments according to similarities in their "
        "TF-IDF features without using the human sentiment labels."
    )

    # Display the selected number of clusters
    st.metric(
        label="Selected number of clusters",
        value=best_cluster_number
    )

    # Display the evaluation measurements
    st.subheader("Selecting the number of clusters")

    st.dataframe(
        clustering_evaluation_df.round(3),
        use_container_width=True
    )

    st.write(
        "The final cluster number was selected using the highest "
        "silhouette score."
    )

    # Create two charts beside each other
    evaluation_figure, evaluation_axes = plt.subplots(
        1,
        2,
        figsize=(12, 4)
    )

    # Plot inertia for the elbow method
    evaluation_axes[0].plot(
        clustering_evaluation_df["Number of clusters"],
        clustering_evaluation_df["Inertia"],
        marker="o"
    )

    evaluation_axes[0].set_xlabel("Number of clusters")
    evaluation_axes[0].set_ylabel("Inertia")
    evaluation_axes[0].set_title("Elbow method")

    # Plot the silhouette scores
    evaluation_axes[1].plot(
        clustering_evaluation_df["Number of clusters"],
        clustering_evaluation_df["Silhouette score"],
        marker="o",
        color="green"
    )

    evaluation_axes[1].set_xlabel("Number of clusters")
    evaluation_axes[1].set_ylabel("Silhouette score")
    evaluation_axes[1].set_title("Silhouette scores")

    # Improve spacing and display the charts
    evaluation_figure.tight_layout()
    st.pyplot(evaluation_figure)
    plt.close(evaluation_figure)

    # Display the number of comments in each cluster
    st.subheader("Cluster sizes")

    st.dataframe(
        cluster_counts,
        use_container_width=True
    )

    # Display the terms that most strongly represent each cluster
    st.subheader("Most important terms in each cluster")

    st.dataframe(
        cluster_terms_df,
        use_container_width=True
    )

    # Create a scatter plot of the comment clusteres
    cluster_figure, cluster_axis = plt.subplots(
        figsize=(9, 6)
    )

    sns.scatterplot(
        data=cluster_plot_data,
        x="Dimension 1",
        y="Dimension 2",
        hue="Cluster",

        # Use different colours for each cluster
        palette="tab10",

        # Make the plotted points easier to see
        alpha=0.7,
        ax=cluster_axis
    )

    cluster_axis.set_title(
        "Two-dimensional visualisation of comment clusters"
    )

    cluster_axis.set_xlabel("Reduced TF-IDF dimension 1")
    cluster_axis.set_ylabel("Reduced TF-IDF dimension 2")

    st.pyplot(cluster_figure)
    plt.close(cluster_figure)

    st.subheader("Clusters compared with human sentiment")

    st.write(
        "The sentiment labels were not used to create the clusters. "
        "They are shown here only to help interpret the results."
    )

    st.dataframe(
        cluster_sentiment_table,
        use_container_width=True
    )

    st.subheader("Example comments from each cluster")

    for cluster_number in range(best_cluster_number):
        st.write(f"**Cluster {cluster_number}**")

        # Select up to five coments assigned to the current cluster
        cluster_examples = cleaned_data[
            cleaned_data["Cluster"] == cluster_number
        ][
            ["Comments", "Sentiment"]
        ].head(5)

        st.dataframe(
            cluster_examples,
            use_container_width=True
        )