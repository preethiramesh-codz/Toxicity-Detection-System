import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("Loading dataset...")

df = pd.read_csv("dataset/train.csv")

df["toxic_label"] = (
    df["toxic"]
    + df["severe_toxic"]
    + df["obscene"]
    + df["threat"]
    + df["insult"]
    + df["identity_hate"]
)

df["toxic_label"] = df["toxic_label"].apply(
    lambda x: 1 if x > 0 else 0
)

X = df["comment_text"]
y = df["toxic_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            max_features=15000
        )
    ),
    (
        "classifier",
        LogisticRegression(max_iter=1000)
    )
])

print("Training Model...")

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print(
    f"Accuracy : {accuracy:.4f}"
)

joblib.dump(
    pipeline,
    "model/toxicity_model.pkl"
)

print("Model Saved Successfully")