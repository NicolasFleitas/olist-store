import numpy as np
import pandas as pd


def haversine(lat1, lon1, lat2, lon2):
    """Calcula la distancia geodésica en kilómetros entre dos pares de coordenadas."""
    r = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def prepare_geolocation(geo: pd.DataFrame) -> pd.DataFrame:
    """Promedia coordenadas por prefijo de código postal."""
    return (
        geo.groupby("geolocation_zip_code_prefix")
        .agg(
            {
                "geolocation_lat": "mean",
                "geolocation_lng": "mean",
            }
        )
        .reset_index()
    )


def build_delivery_features(
    orders: pd.DataFrame,
    items: pd.DataFrame,
    products: pd.DataFrame,
    customers: pd.DataFrame,
    sellers: pd.DataFrame,
    geo: pd.DataFrame,
) -> pd.DataFrame:
    """Construye el dataset unificado con ingeniería de features para predicción de entregas."""
    geo_coords = prepare_geolocation(geo)

    products = products.copy()
    products["product_volume_cm3"] = (
        products["product_length_cm"] * products["product_height_cm"] * products["product_width_cm"]
    )

    # Filtrado de órdenes con un único vendedor
    seller_counts = items.groupby("order_id")["seller_id"].nunique()
    valid_order_ids = [order_id for order_id, count in seller_counts.items() if count == 1]

    # Filtrar items para quedarnos SOLO con ordenes que tengan un solo vendedor
    items_single_seller = items[items["order_id"].isin(valid_order_ids)]

    # Agregar datos de productos a items
    items_single_seller = items_single_seller.merge(
        products[["product_id", "product_weight_g", "product_volume_cm3"]], on="product_id"
    )

    # Resumir datos de items por orden
    orders_items_agg = (
        items_single_seller.groupby(["order_id", "seller_id"])
        .agg(
            total_weight_g=("product_weight_g", "sum"),
            total_volume_cm3=("product_volume_cm3", "sum"),
            total_freight=("freight_value", "sum"),
            n_items=("order_item_id", "count"),
        )
        .reset_index()
    )

    # Merges y Distancia
    df = orders.merge(orders_items_agg, on="order_id")
    df = df.merge(
        customers[["customer_id", "customer_zip_code_prefix", "customer_state"]],
        on="customer_id",
    )
    df = df.merge(
        sellers[["seller_id", "seller_zip_code_prefix", "seller_state"]],
        on="seller_id",
    )

    df = (
        df.merge(
            geo_coords,
            left_on="customer_zip_code_prefix",
            right_on="geolocation_zip_code_prefix",
            how="left",
        )
        .rename(columns={"geolocation_lat": "cust_lat", "geolocation_lng": "cust_lng"})
        .drop(columns="geolocation_zip_code_prefix")
    )

    df = (
        df.merge(
            geo_coords,
            left_on="seller_zip_code_prefix",
            right_on="geolocation_zip_code_prefix",
            how="left",
        )
        .rename(columns={"geolocation_lat": "sel_lat", "geolocation_lng": "sel_lng"})
        .drop(columns="geolocation_zip_code_prefix")
    )

    # Manejo de nulos en coordenadas antes de aplicar Haversine
    df = df.dropna(subset=["cust_lat", "cust_lng", "sel_lat", "sel_lng"])

    # Creación de caracteristica de distancia
    df["distance_km"] = haversine(df["cust_lat"], df["cust_lng"], df["sel_lat"], df["sel_lng"])

    # Variable de logística estatal (mismo estado vs interestadual)
    df["is_same_state"] = (df["customer_state"] == df["seller_state"]).astype(int)

    # Fechas, Estacionalidad y Target
    date_cols = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Filtrar solo pedidos completados y con fechas no nulas
    df = df[df["order_status"] == "delivered"]
    assert isinstance(df, pd.DataFrame)
    df = df.dropna(subset=["order_purchase_timestamp", "order_delivered_customer_date"])

    df["order_month"] = df["order_purchase_timestamp"].dt.month

    # Filtrar solo los casos válidos (entrega >= compra)
    df = df[df["order_delivered_customer_date"] >= df["order_purchase_timestamp"]].copy()
    assert isinstance(df, pd.DataFrame)

    # Cálculo del tiempo total de entrega en días (variable target)
    delivery_delta = df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    if isinstance(delivery_delta, pd.Series):
        df["target_days"] = delivery_delta.dt.total_seconds() / 86400

    # Cálculo del tiempo estimado de entrega prometido al cliente
    estimated_delta = df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]
    if isinstance(estimated_delta, pd.Series):
        df["estimated_days"] = estimated_delta.dt.total_seconds() / 86400

    return df


def clean_outliers(df: pd.DataFrame, quantile: float = 0.99) -> pd.DataFrame:
    """Filtra los pedidos cuyo target_days supera el percentil indicado (por defecto 0.99)."""
    limit_days = df["target_days"].quantile(quantile)
    filtered = df[df["target_days"] <= limit_days]
    assert isinstance(filtered, pd.DataFrame)
    return filtered.copy()
