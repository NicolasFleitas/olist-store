"""Features y modelado de clustering para sistemas de recomendación."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURE_COLS: list[str] = [
    "log_price",
    "log_freight",
    "freight_ratio",
    "review_score",
    "log_sales",
]


def build_product_features(
    products: pd.DataFrame,
    items: pd.DataFrame,
    reviews: pd.DataFrame,
) -> pd.DataFrame:
    """Construye el dataset consolidado de métricas por producto para recomendación."""
    # Unir items con reviews por order_id
    items_reviews = items.merge(reviews[["order_id", "review_score"]], on="order_id", how="left")

    # Agregación a nivel de producto
    prod_stats = (
        items_reviews.groupby("product_id")
        .agg(
            price=("price", "mean"),
            freight_value=("freight_value", "mean"),
            total_sales=("order_item_id", "count"),
            review_score=("review_score", "mean"),
        )
        .reset_index()
    )
    assert isinstance(prod_stats, pd.DataFrame)

    # Imputación de review_score faltantes con la media global
    global_review_mean = float(np.mean(prod_stats["review_score"]))
    prod_stats["review_score"] = prod_stats["review_score"].fillna(global_review_mean)

    # Ratio de flete sobre precio con recorte al percentil 99.9
    prod_stats["freight_ratio"] = prod_stats["freight_value"] / (prod_stats["price"] + 1e-5)
    q999 = float(np.quantile(prod_stats["freight_ratio"], 0.999))
    prod_stats["freight_ratio"] = prod_stats["freight_ratio"].clip(upper=q999)

    # Transformación logarítmica para estabilizar variables asimétricas
    prod_stats["log_price"] = np.log1p(prod_stats["price"])
    prod_stats["log_freight"] = np.log1p(prod_stats["freight_value"])
    prod_stats["log_sales"] = np.log1p(prod_stats["total_sales"])

    # Unir con la información de categoría del producto
    df_model = prod_stats.merge(
        products[["product_id", "product_category_name"]], on="product_id", how="left"
    )
    assert isinstance(df_model, pd.DataFrame)
    df_model["product_category_name"] = df_model["product_category_name"].fillna("unknown")

    return df_model


def scale_features(
    df_model: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, StandardScaler]:
    """Estandariza las características de los productos usando StandardScaler."""
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_model[feature_cols])
    return np.asarray(scaled), scaler


def compute_pca(
    scaled_features: np.ndarray,
    n_components: int = 2,
    random_state: int = 42,
) -> tuple[np.ndarray, PCA]:
    """Proyecta las variables multidimensionales a un espacio 2D mediante PCA."""
    pca = PCA(n_components=n_components, random_state=random_state)
    pca_data = pca.fit_transform(scaled_features)
    return np.asarray(pca_data), pca


def find_optimal_clusters(
    scaled_features: np.ndarray,
    k_range: list[int] | range = range(2, 8),
    sample_size: int = 10000,
    random_state: int = 42,
) -> tuple[list[float], list[float], dict[int, float], dict[int, float], int]:
    """Evalúa el rango de clusters con Método del Codo y Silhouette Score."""
    inertias: list[float] = []
    silhouettes: list[float] = []
    inertia_vals: dict[int, float] = {}
    sil_vals: dict[int, float] = {}

    sample_size_actual = min(sample_size, len(scaled_features))
    sample_indices = np.random.RandomState(random_state).choice(
        len(scaled_features), size=sample_size_actual, replace=False
    )

    for k in k_range:
        kmeans = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=10,  # pyright: ignore[reportArgumentType]
        )
        labels = kmeans.fit_predict(scaled_features)
        inertia_score = float(kmeans.inertia_)  # pyright: ignore[reportArgumentType]
        inertias.append(inertia_score)
        inertia_vals[k] = inertia_score
        sil = float(silhouette_score(scaled_features[sample_indices], labels[sample_indices]))
        silhouettes.append(sil)
        sil_vals[k] = sil

    best_k = max(list(sil_vals.keys()), key=lambda k: sil_vals[k])
    return inertias, silhouettes, inertia_vals, sil_vals, best_k


def assign_cluster_names(profile: pd.DataFrame) -> dict[int, str]:
    """Asigna nombres semánticos de negocio a los clusters según sus centroides."""
    names: dict[int, str] = {}
    for c in profile.index:
        cluster_id = int(str(c))
        rev = float(profile.loc[c, "review_score"])  # pyright: ignore[reportArgumentType]
        sales = float(profile.loc[c, "total_sales"])  # pyright: ignore[reportArgumentType]
        ratio = float(profile.loc[c, "freight_ratio"])  # pyright: ignore[reportArgumentType]
        price = float(profile.loc[c, "price"])  # pyright: ignore[reportArgumentType]
        if rev < 2.5:
            names[cluster_id] = "Baja Calificación / Riesgo"
        elif sales > 8.0:
            names[cluster_id] = "Alta Rotación / Best Sellers"
        elif ratio > 0.6:
            names[cluster_id] = "Ultra-Económico / Flete Crítico"
        elif price > 250:
            names[cluster_id] = "Premium / High-Ticket"
        else:
            names[cluster_id] = "Gama Media / Satisfecho"
    return names


def train_kmeans_clusters(
    df_model: pd.DataFrame,
    scaled_features: np.ndarray,
    n_clusters: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, KMeans]:
    """Entrena KMeans, calcula perfiles estadísticos y asigna etiquetas semánticas."""
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,  # pyright: ignore[reportArgumentType]
    )
    df_result = df_model.copy()
    df_result["cluster"] = kmeans.fit_predict(scaled_features)

    # Perfil estadístico real de cada cluster
    mean_profile = df_result.groupby("cluster")[
        ["price", "freight_value", "freight_ratio", "review_score", "total_sales"]
    ].mean()
    assert isinstance(mean_profile, pd.DataFrame)

    cluster_profile = mean_profile.copy()
    counts = df_result["cluster"].value_counts()
    cluster_profile["count"] = counts
    cluster_profile["pct"] = (counts / len(df_result)) * 100

    cluster_names = assign_cluster_names(cluster_profile)
    df_result["cluster_name"] = [
        cluster_names.get(int(str(c)), "Desconocido") for c in df_result["cluster"]
    ]
    cluster_profile["cluster_name"] = [
        cluster_names.get(int(str(idx)), "Desconocido") for idx in cluster_profile.index
    ]

    return df_result, cluster_profile, kmeans


def recomendar_productos(
    product_id: str,
    df: pd.DataFrame,
    features_matrix: np.ndarray,
    n_recs: int = 5,
) -> pd.DataFrame:
    """Recomienda los N productos más similares de la misma categoría y gama."""
    matches = df[df["product_id"] == product_id]
    assert isinstance(matches, pd.DataFrame)
    if matches.empty:
        raise ValueError(f"Producto {product_id} no encontrado en la base de datos.")

    prod_idx = int(str(matches.index[0]))
    category = str(df.loc[prod_idx, "product_category_name"])
    cluster = int(str(df.loc[prod_idx, "cluster"]))
    cluster_name = str(df.loc[prod_idx, "cluster_name"])
    price_val = float(df.loc[prod_idx, "price"])  # pyright: ignore[reportArgumentType]
    freight_val = float(df.loc[prod_idx, "freight_value"])  # pyright: ignore[reportArgumentType]
    review_val = float(df.loc[prod_idx, "review_score"])  # pyright: ignore[reportArgumentType]
    sales_val = int(df.loc[prod_idx, "total_sales"])  # pyright: ignore[reportArgumentType]

    # 1. Candidatos: Misma categoría y mismo cluster (excluyendo producto consultado)
    same_cat_cluster = df[
        (df["product_category_name"] == category)
        & (df["cluster"] == cluster)
        & (df["product_id"] != product_id)
    ]
    assert isinstance(same_cat_cluster, pd.DataFrame)

    # 2. Fallback: Completar con la misma categoría en otros clusters si faltan
    if len(same_cat_cluster) < n_recs:
        other_clusters = df[
            (df["product_category_name"] == category)
            & (df["cluster"] != cluster)
            & (df["product_id"] != product_id)
        ]
        assert isinstance(other_clusters, pd.DataFrame)
        cands = pd.concat([same_cat_cluster, other_clusters])
    else:
        cands = same_cat_cluster

    assert isinstance(cands, pd.DataFrame)
    if cands.empty:
        return pd.DataFrame(
            columns=[
                "product_id",
                "product_category_name",
                "cluster_name",
                "price",
                "freight_value",
                "review_score",
                "distancia_similitud",
            ]
        )

    # 3. Calcular distancia euclidiana en el espacio multidimensional escalado
    cand_indices = [int(str(i)) for i in cands.index]
    distances = np.linalg.norm(features_matrix[cand_indices] - features_matrix[prod_idx], axis=1)

    cands = cands.copy()
    cands["distancia_similitud"] = distances

    recommendations = cands.sort_values(by="distancia_similitud").head(n_recs)
    assert isinstance(recommendations, pd.DataFrame)

    print("=" * 80)
    print(f"PRODUCTO CONSULTADO: {product_id}")
    print(f"Categoría: {category} | Gama/Cluster: {cluster_name}")
    print(
        f"Precio: ${price_val:.2f} | "
        f"Flete: ${freight_val:.2f} | "
        f"Review: {review_val:.1f}⭐ | "
        f"Ventas: {sales_val}"
    )
    print("=" * 80)
    print("PRODUCTOS RECOMENDADOS (Misma categoría y gama similar):")

    output_cols = [
        "product_id",
        "product_category_name",
        "cluster_name",
        "price",
        "freight_value",
        "review_score",
        "distancia_similitud",
    ]
    res = recommendations[output_cols].reset_index(drop=True)
    assert isinstance(res, pd.DataFrame)
    return res


def evaluate_clustering_metrics(
    scaled_features: np.ndarray,
    cluster_labels: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Calcula los índices Calinski-Harabasz y Davies-Bouldin del clustering."""
    ch_idx = float(calinski_harabasz_score(scaled_features, cluster_labels))
    db_idx = float(davies_bouldin_score(scaled_features, cluster_labels))
    return {
        "calinski_harabasz_score": ch_idx,
        "davies_bouldin_score": db_idx,
    }
