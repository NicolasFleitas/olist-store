from pathlib import Path

import pandas as pd

from src.common.config import DATA


def load_csv(filename: str) -> pd.DataFrame:
    """Carga un archivo CSV desde el directorio común de datasets."""
    return pd.read_csv(DATA / filename)


def list_datasets() -> list[Path]:
    """Lista las rutas de todos los archivos CSV disponibles en datasets."""
    return list(DATA.glob("*.csv"))


def load_delivery_raw_data() -> dict[str, pd.DataFrame]:
    """Carga las 6 tablas relacionales base para el modelo de tiempos de entrega."""
    return {
        "orders": load_csv("olist_orders_dataset.csv"),
        "items": load_csv("olist_order_items_dataset.csv"),
        "products": load_csv("olist_products_dataset.csv"),
        "customers": load_csv("olist_customers_dataset.csv"),
        "sellers": load_csv("olist_sellers_dataset.csv"),
        "geo": load_csv("olist_geolocation_dataset.csv"),
    }


def load_sentiment_raw_data() -> pd.DataFrame:
    """Carga la tabla de reseñas de órdenes para análisis de sentimiento."""
    return load_csv("olist_order_reviews_dataset.csv")
