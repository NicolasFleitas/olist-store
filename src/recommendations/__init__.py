"""Módulo de recomendaciones de productos."""

from src.recommendations.features import (
    FEATURE_COLS,
    assign_cluster_names,
    build_product_features,
    compute_pca,
    evaluate_clustering_metrics,
    find_optimal_clusters,
    recomendar_productos,
    scale_features,
    train_kmeans_clusters,
)
from src.recommendations.viz import (
    plot_clusters_pca_2d,
    plot_elbow_and_silhouette,
)

__all__ = [
    "FEATURE_COLS",
    "assign_cluster_names",
    "build_product_features",
    "compute_pca",
    "evaluate_clustering_metrics",
    "find_optimal_clusters",
    "recomendar_productos",
    "scale_features",
    "train_kmeans_clusters",
    "plot_clusters_pca_2d",
    "plot_elbow_and_silhouette",
]
