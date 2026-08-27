"""Visualizaciones para análisis de sentimiento y procesamiento de texto."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline


def plot_confusion_matrices(
    pipelines: dict[str, Pipeline],
    X_test: pd.Series,
    y_test: pd.Series,
) -> None:
    """Grafica matrices de confusión para cada pipeline evaluado."""
    num_models = len(pipelines)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 4))
    if num_models == 1:
        axes = np.array([axes])

    for ax, (nombre, pipe) in zip(axes, pipelines.items(), strict=True):
        ConfusionMatrixDisplay.from_estimator(
            pipe, X_test, y_test, display_labels=["Negativo", "Positivo"], cmap="Blues", ax=ax
        )
        ax.set_title(f"Matriz de Confusión: {nombre}")

    plt.tight_layout()
    plt.show()


def plot_top_coefficients(
    pipeline: Pipeline,
    top_n: int = 15,
) -> tuple[pd.Series, pd.Series]:
    """Extrae y grafica las palabras con mayor peso positivo y negativo en un modelo lineal."""
    vectorizer = pipeline.named_steps["tfidf"]
    model = pipeline.named_steps["clf"]
    features = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]

    top_idx = coefs.argsort()
    neg = pd.Series(coefs[top_idx[:top_n]], index=features[top_idx[:top_n]])
    pos = pd.Series(coefs[top_idx[-top_n:]], index=features[top_idx[-top_n:]])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    data_plan = zip(
        axes,
        [neg, pos],
        ["NEGATIVO", "POSITIVO"],
        ["#d62728", "#2ca02c"],
        strict=True,
    )

    for ax, series, title, color in data_plan:
        series.plot.barh(ax=ax, color=color).invert_yaxis()
        ax.set_title(f"Top {top_n} Palabras que Indican {title}", fontweight="bold", fontsize=12)
        ax.set_xlabel("Coeficiente (importancia)", fontsize=11)
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.show()
    return neg, pos
