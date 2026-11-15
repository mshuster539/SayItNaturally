
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


# ==========================================
# 1. LOAD AND PREPARE THE DATASETS
# ==========================================

formal = pd.read_csv("data/formal_cleaned.csv")
informal = pd.read_csv("data/informal_cleaned.csv")

print("Formal dataset:", formal.shape)
print("Informal dataset:", informal.shape)

# Assign register labels
formal["register"] = "formal"
informal["register"] = "informal"

# Combine datasets
data = pd.concat([formal, informal], ignore_index=True)

print("\nCombined dataset:", data.shape)
print(data.head())


# ==========================================
# 2. VISUALISE DATASET DISTRIBUTION
# ==========================================

register_counts = data["register"].value_counts()

plt.figure(figsize=(7, 5))
register_counts.plot(kind="bar")

plt.title("Distribution of Formal and Informal Sentences")
plt.xlabel("Register")
plt.ylabel("Number of Sentences")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# ==========================================
# 3. EXPLORE SENTENCE LENGTH
# ==========================================

data["sentence_length"] = data["English"].str.len()

print("\nAverage sentence length:")
print(data.groupby("register")["sentence_length"].mean())


# ==========================================
# 4. SPLIT THE DATA
# ==========================================

X = data["English"]
y = data["register"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining examples:", len(X_train))
print("Testing examples:", len(X_test))


# ==========================================
# 5. TRAIN LOGISTIC REGRESSION
# ==========================================

model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), min_df=2),
    LogisticRegression(max_iter=1000)
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"\nLogistic Regression accuracy: {accuracy:.1%}")


# ==========================================
# 6. BASELINE MODEL
# ==========================================

baseline = DummyClassifier(strategy="most_frequent")

baseline.fit(X_train.to_frame(), y_train)

baseline_predictions = baseline.predict(X_test.to_frame())

baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions
)

print(f"\nBaseline accuracy: {baseline_accuracy:.1%}")
print(f"Logistic Regression accuracy: {accuracy:.1%}")


# ==========================================
# 7. MODEL ACCURACY COMPARISON
# ==========================================

models = ["Baseline", "Logistic Regression"]
accuracies = [baseline_accuracy, accuracy]

plt.figure(figsize=(7, 5))
plt.bar(models, accuracies)

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)

for i, value in enumerate(accuracies):
    plt.text(i, value + 0.02, f"{value:.1%}", ha="center")

plt.tight_layout()
plt.show()


# ==========================================
# 8. CONFUSION MATRIX
# ==========================================

ConfusionMatrixDisplay.from_estimator(
    model,
    X_test,
    y_test,
    display_labels=["Formal", "Informal"],
    cmap="Blues"
)

plt.title("Where Does the Classifier Make Mistakes?")
plt.tight_layout()
plt.show()


# ==========================================
# 9. CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:")

print(classification_report(
    y_test,
    predictions,
    labels=["formal", "informal"],
    target_names=["Formal", "Informal"]
))


# ==========================================
# 10. DISPLAY SAMPLE PREDICTIONS
# ==========================================

results = pd.DataFrame({
    "English sentence": X_test,
    "Actual register": y_test,
    "Predicted register": predictions
})

print("\nSample predictions:")
print(results.sample(10, random_state=42).to_string(index=False))


# ==========================================
# 11. TEST OUR OWN EXAMPLES
# ==========================================

custom_data = pd.read_csv("data/my_own_examples.csv")

print("\nCustom dataset:")
print(custom_data.head())

print("\nCustom register distribution:")
print(custom_data["register"].value_counts())

custom_predictions = model.predict(
    custom_data["natural_english"]
)

custom_results = pd.DataFrame({
    "English sentence": custom_data["natural_english"],
    "Actual register": custom_data["register"],
    "Predicted register": custom_predictions
})

print("\nCustom evaluation results:")
print(custom_results.to_string(index=False))

print("\nCustom classification report:")

print(classification_report(
    custom_data["register"],
    custom_predictions,
    labels=["formal", "informal"],
    target_names=["Formal", "Informal"]
))


# ==========================================
# 12. SAVE THE TRAINED CLASSIFIER
# ==========================================

Path("models").mkdir(exist_ok=True)

joblib.dump(
    model,
    "models/register_classifier.joblib"
)

print("\nClassifier saved successfully!")
