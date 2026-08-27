"""Módulo de análisis de sentimiento y procesamiento de lenguaje natural."""

from src.sentiment.features import (
    build_sentiment_features,
    build_sentiment_pipelines,
    evaluate_sentiment_models,
    get_portuguese_stopwords,
    remove_accents,
)
from src.sentiment.viz import plot_confusion_matrices, plot_top_coefficients

__all__ = [
    "build_sentiment_features",
    "build_sentiment_pipelines",
    "evaluate_sentiment_models",
    "get_portuguese_stopwords",
    "remove_accents",
    "plot_confusion_matrices",
    "plot_top_coefficients",
]
