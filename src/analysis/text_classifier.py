"""Lab 5 Part 1: a trainable sentiment/text classifier.

TF-IDF features + Logistic Regression in an sklearn Pipeline, with an optional
RandomizedSearchCV hyper-parameter sweep and a classification_report — the exact
Lab 5 supervised-learning workflow. Train it on any labeled review dataset
(e.g. Restaurant_Reviews.tsv) and reuse it on the YouTube comments.
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


def train(
    texts: list[str],
    labels: list,
    tune: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Train the classifier; returns (fitted_model, report_str)."""
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    model: Pipeline | RandomizedSearchCV = build_pipeline()
    if tune:
        from scipy.stats import uniform

        param_dist = {
            "tfidf__max_df": uniform(0.7, 0.3),
            "tfidf__min_df": [1, 2, 5],
            "clf__C": uniform(0.1, 10),
        }
        model = RandomizedSearchCV(
            build_pipeline(), param_dist, n_iter=10, cv=3, random_state=random_state
        )

    model.fit(X_train, y_train)
    report = classification_report(y_test, model.predict(X_test))
    return model, report
