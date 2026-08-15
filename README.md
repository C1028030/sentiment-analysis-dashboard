# YouTube Comment Analysis Dashboard

A Streamlit application that analyses the sentiment and similarity of YouTube comments using natural language processing, supervised machine learning, rule-based sentiment analysis, unsupervised learning and a pretrained transformer.

## Features

- Loads, validates and cleans a labelled YouTube-comment dataset.
- Preprocesses comment text for machine-learning analysis.
- Converts text into numerical features using TF-IDF.
- Classifies comments with Logistic Regression and Linear SVM.
- Analyses sentiment using VADER's rule-based approach.
- Groups similar comments using K-means clustering.
- Uses a pretrained RoBERTa model for transformer-based sentiment analysis.
- Compares models using accuracy, precision, recall, F1-score and confusion matrices.
- Provides live analysis for new comments entered by a user.
- Displays results across separate Streamlit dashboard tabs.

## AI and NLP Methods

### TF-IDF

TF-IDF converts processed comments into numerical features. The vectoriser is fitted only on the training data for supervised classification, preventing information from the test set from leaking into model training.

### Logistic Regression

Logistic Regression provides a supervised sentiment-classification baseline. It learns relationships between TF-IDF features and the human-provided negative, neutral and positive labels.

### Linear SVM

Linear Support Vector Machine is trained using the same training set and TF-IDF features as Logistic Regression. Using the same unseen test set allows the classifiers to be compared fairly.

### VADER

VADER is a lexicon-based, rule-based sentiment analyser. It does not learn from the project dataset. It evaluates features such as sentiment-bearing words, punctuation and capitalisation to produce a compound score and sentiment label.

### K-means Clustering

K-means is an unsupervised technique that groups comments by similarity without using the human sentiment labels. Candidate cluster counts are assessed using inertia and silhouette scores. Top TF-IDF terms and example comments help interpret each resulting cluster.

### Pretrained RoBERTa

The advanced feature uses [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), a pretrained transformer designed for English social-media sentiment analysis. It predicts negative, neutral or positive sentiment and supplies a confidence score.

## Dashboard Tabs

1. **Live Analysis** – compares predictions for a new comment using Logistic Regression, Linear SVM, VADER, RoBERTa and K-means.
2. **Data and NLP** – displays data-cleaning evidence, label distribution, preprocessing examples and TF-IDF information.
3. **Classifiers** – displays classifier reports, confusion matrices, RoBERTa evaluation and the model-comparison table.
4. **VADER** – displays VADER performance, its confusion matrix and example predictions.
5. **Clustering** – displays cluster selection, cluster sizes, important terms, visualisation and example comments.

## Project Structure

```text
sentiment-analysis-dashboard/
|-- app.py
|-- requirements.txt
|-- README.md
|-- .gitignore
`-- data/
    `-- youtube_comments_labels.csv
```

The dataset must contain these columns:

| Column | Purpose |
|---|---|
| `Comments` | Original YouTube comment text |
| `Sentiment` | Human label: `negative`, `neutral` or `positive` |

## Installation

### 1. Clone the repository

```powershell
git clone https://github.com/C1028030/sentiment-analysis-dashboard.git
cd sentiment-analysis-dashboard
```

Replace `YOUR-USERNAME` with the repository owner's GitHub username.

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

### 3. Activate the virtual environment

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once in the current terminal and try again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 4. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run the application

```powershell
streamlit run app.py
```

Streamlit will display a local address, normally:

```text
http://localhost:8501
```

## Using the Application

1. Open the **Live Analysis** tab.
2. Enter a YouTube comment.
3. Select **Analyse comment**.
4. Review the sentiment predictions, RoBERTa confidence and K-means cluster.
5. Use the other tabs to inspect preprocessing, evaluation and clustering evidence.
6. In the **Classifiers** tab, select **Evaluate RoBERTa on test set** to include it in the model-comparison table.

The first RoBERTa request may take longer because the pretrained model must be downloaded and cached. Later predictions should be faster.

## Data Processing

The application:

- Checks that the required CSV columns exist.
- Removes rows with missing comments or labels.
- Removes empty and duplicate comments.
- Standardises sentiment labels to lowercase.
- Retains only negative, neutral and positive labels.
- Creates a separate processed-comment column while preserving the original text.
- Splits labelled data into stratified 80% training and 20% testing sets.
- Fits the classification TF-IDF vectoriser only on training comments.

## Evaluation

The sentiment methods are evaluated against the same human-labelled test comments using:

- Accuracy
- Precision
- Recall
- F1-score
- Macro F1-score
- Confusion matrices

Macro F1 gives each sentiment class equal importance, making it useful when class frequencies are uneven. K-means is evaluated separately using inertia and silhouette scores because clustering does not predict the human sentiment labels directly.

## Input Validation and Error Handling

- Empty comments are rejected.
- Comments containing no usable processed text are rejected.
- Live input is limited to 5,000 characters.
- Users are warned when no input terms appear in the TF-IDF training vocabulary.
- Transformer input is truncated to its supported token limit.
- Missing, empty or incorrectly structured datasets produce clear error messages.
- If RoBERTa is unavailable, the other analysis methods remain usable.

## Limitations

- Predictions depend on the size, quality and balance of the labelled dataset.
- TF-IDF does not fully understand context, word order or sarcasm.
- VADER can struggle with sarcasm, specialist vocabulary and context-dependent language.
- RoBERTa was pretrained on social-media data and may still produce confident but incorrect predictions.
- Transformer inference is slower and requires more memory than the other methods.
- K-means cluster numbers are identifiers and do not automatically represent sentiment categories.
- The two-dimensional cluster plot is a reduced visual representation and does not preserve every feature from the original TF-IDF space.

## Reproducibility

Fixed `random_state` values are used for the train/test split and machine-learning models where applicable. The same test set is used to compare Logistic Regression, Linear SVM, VADER and RoBERTa.

## Academic Use

This project was developed for the Programming for Artificial Intelligence module. Any use of AI-assisted development should be disclosed in accordance with the relevant assessment brief and university academic-conduct requirements.
