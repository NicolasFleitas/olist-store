"""Visualizaciones diagnósticas para sistemas de recomendación."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_elbow_and_silhouette(
    k_range: list[int] | range,
    inertias: list[float],
    silhouettes: list[float],
) -> None:
    """Grafica el Método del Codo y el Silhouette Score para selección de K."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(list(k_range), inertias, "bo-", linewidth=2, markersize=6)
    ax[0].set_title("Método del Codo (Inertia en 5D)")
    ax[0].set_xlabel("Número de Clusters (K)")
    ax[0].set_ylabel("Inertia")
    ax[0].grid(True)

    ax[1].plot(list(k_range), silhouettes, "ro-", linewidth=2, markersize=6)
    ax[1].set_title("Silhouette Score por K (en 5D)")
    ax[1].set_xlabel("Número de Clusters (K)")
    ax[1].set_ylabel("Silhouette Score")
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_clusters_pca_2d(
    pca_data: np.ndarray,
    cluster_names: pd.Series | list[str],
) -> None:
    """Visualiza la dispersión de los clusters en el espacio proyectado 2D de PCA."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=pca_data[:, 0],
        y=pca_data[:, 1],
        hue=cluster_names,
        palette="tab10",
        alpha=0.6,
        s=15,
    )
    plt.title("Visualización de Clusters en Espacio PCA (2D)")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
