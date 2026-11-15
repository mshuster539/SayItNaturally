import joblib
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier

# Find the main project folder
project_folder = Path(__file__).resolve().parent.parent

# Define the data and models folders
data_folder = project_folder / "data"
models_folder = project_folder / "models"


# Load the cleaned formal and informal datasets
formal = pd.read_csv(data_folder / "formal_cleaned.csv")
informal = pd.read_csv(data_folder / "informal_cleaned.csv")

# Assign a register label to each dataset
formal["register"] = "formal"
informal["register"] = "informal"

# Combine both datasets into one
data = pd.concat([formal, informal], ignore_index=True)


# VISUALIZATION 1: Distribution of formal and informal sentences
register_counts = data["register"].value_counts()

plt.figure(figsize=(7, 5))
register_counts.plot(kind="bar")

plt.title("Distribution of Formal and Informal Sentences")
plt.xlabel("Register")
plt.ylabel("Number of Sentences")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# Separate the English sentences and their register labels
X = data["English"]
y = data["register"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Create a machine learning pipeline
# TF-IDF converts sentences into numerical features
# Logistic Regression learns to classify their register
model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), min_df=2),
    LogisticRegression(max_iter=1000)
)

# Train the classifier using the training data
model.fit(X_train, y_train)

# Predict the register of the test sentences
predictions = model.predict(X_test)

# Calculate the classifier's accuracy
accuracy = accuracy_score(y_test, predictions)


# Train a baseline model for comparison
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train.to_frame(), y_train)

baseline_predictions = baseline.predict(X_test.to_frame())

baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions
)


# VISUALIZATION 2: Compare the accuracy of both models
models = ["Baseline", "Logistic Regression"]
accuracies = [baseline_accuracy, accuracy]

plt.figure(figsize=(7, 5))
plt.bar(models, accuracies)

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1.1)

for i, value in enumerate(accuracies):
    plt.text(i, value + 0.02, f"{value:.1%}", ha="center")

plt.tight_layout()
plt.show()


# VISUALIZATION 3: Display a confusion matrix
ConfusionMatrixDisplay.from_estimator(
    model,
    X_test,
    y_test,
    display_labels=["Formal", "Informal"],
    cmap="Blues"
)

plt.title("Classifier Confusion Matrix")
plt.tight_layout()
plt.show()


# Evaluate the classifier using precision, recall and F1-score
print("Classification Report:")

print(classification_report(
    y_test,
    predictions,
    labels=["formal", "informal"],
    target_names=["Formal", "Informal"],
    zero_division=0
))


# Load my own examples for additional evaluation
custom_data = pd.read_csv(data_folder / "my_own_examples.csv")

# Predict the register of my own English sentences
custom_predictions = model.predict(
    custom_data["natural_english"]
)

# Compare the actual and predicted register
custom_results = pd.DataFrame({
    "English sentence": custom_data["natural_english"],
    "Actual register": custom_data["register"],
    "Predicted register": custom_predictions
})

print("Custom evaluation results:")
print(custom_results.to_string(index=False))


# Evaluate the classifier on my own examples
print("Custom Classification Report:")

print(classification_report(
    custom_data["register"],
    custom_predictions,
    labels=["formal", "informal"],
    target_names=["Formal", "Informal"],
    zero_division=0
))


# Save the trained classifier for use in the application
models_folder.mkdir(exist_ok=True)

joblib.dump(
    model,
    models_folder / "register_classifier.joblib"
)