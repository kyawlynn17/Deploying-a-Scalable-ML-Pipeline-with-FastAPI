import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ml.data import apply_label
from ml.model import (
    compute_model_metrics,
    inference,
    train_model,
)


def test_apply_labels():
    assert apply_label(np.array([1])) == ">50K"
    assert apply_label(np.array([0])) == "<=50K"


def test_train_model():
    X = np.array(
        [
            [25, 40],
            [45, 50],
            [30, 35],
            [55, 60],
        ]
    )
    y = np.array([0, 1, 0, 1])

    model = train_model(X, y)

    assert isinstance(model, RandomForestClassifier)


def test_inference():
    X = np.array(
        [
            [25, 40],
            [45, 50],
            [30, 35],
            [55, 60],
        ]
    )
    y = np.array([0, 1, 0, 1])

    model = train_model(X, y)
    preds = inference(model, X)

    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(y)


def test_compute_model_metrics():
    y = np.array([0, 1, 1, 0])
    preds = np.array([0, 1, 1, 0])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert precision == 1.0
    assert recall == 1.0
    assert fbeta == 1.0