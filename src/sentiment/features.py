"""Features y procesamiento de texto para análisis de sentimiento."""

import unicodedata

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def remove_accents(s: str) -> str:
    """Normaliza acentos de un texto para alinearlo con strip_accents='unicode'."""
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def get_portuguese_stopwords(exclude_negations: bool = True) -> list[str]:
    """Obtiene y normaliza stopwords en portugués, conservando opcionalmente negaciones."""
    nltk.download("stopwords", quiet=True)
    stop_words = {remove_accents(w) for w in stopwords.words("portuguese")}
    if exclude_negations:
        negacoes = {
            "nao",
            "nunca",
            "jamais",
            "nenhum",
            "nenhuma",
            "nem",
            "nada",
            "sem",
            "tampouco",
        }
        return sorted(list(stop_words - negacoes))
    return sorted(list(stop_words))


def build_sentiment_features(df_reviews: pd.DataFrame) -> pd.DataFrame:
    """Limpia y construye el dataset procesado para análisis de sentimiento."""
    # 1. Conservar registros con algún texto (título y/o mensaje)
    df_text = df_reviews.dropna(
        subset=["review_comment_title", "review_comment_message"], how="all"
    ).copy()

    # 2. Excluir reviews neutras (3 estrellas) y definir target binario (>=4 -> 1, <=2 -> 0)
    df_filtered = df_text[df_text["review_score"] != 3].copy()
    assert isinstance(df_filtered, pd.DataFrame)
    df_filtered["target"] = (df_filtered["review_score"] >= 4).astype(int)

    # 3. Ordenar cronológicamente por fecha de creación de la review
    df_filtered["review_creation_date"] = pd.to_datetime(df_filtered["review_creation_date"])
    df_sorted = df_filtered.sort_values(by="review_creation_date").reset_index(drop=True)

    # 4. Construir review_text_full combinando título + mensaje
    title = df_sorted["review_comment_title"].fillna("").astype(str).str.strip().str.rstrip(".!?:;")

    message = df_sorted["review_comment_message"].fillna("").astype(str).str.strip()
    sep = np.where((title != "") & (message != ""), ". ", "")
    df_sorted["review_text_full"] = (
        (title + sep + message).str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
    )

    return df_sorted


def build_sentiment_pipelines(
    stop_words: list[str] | None = None,
) -> dict[str, Pipeline]:
    """Construye los pipelines comparativos de Naive Bayes y Linear SVM."""
    if stop_words is None:
        stop_words = get_portuguese_stopwords(exclude_negations=True)

    return {
        "Naive Bayes (MultinomialNB)": Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        strip_accents="unicode",
                        ngram_range=(1, 2),
                        min_df=2,
                        stop_words=stop_words,
                    ),
                ),
                ("clf", MultinomialNB()),
            ]
        ),
        "Linear SVM (Balanced)": Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        strip_accents="unicode",
                        ngram_range=(1, 2),
                        min_df=2,
                        stop_words=stop_words,
                    ),
                ),
                ("clf", LinearSVC(random_state=42, class_weight="balanced")),
            ]
        ),
    }


def evaluate_sentiment_models(
    pipelines: dict[str, Pipeline],
    X_train: pd.Series,
    y_train: pd.Series,
    X_test: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Entrena y evalúa los pipelines de sentimiento, devolviendo métricas comparativas."""
    resultados = []

    for nombre, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        clf = pipe.named_steps["clf"]
        if hasattr(clf, "predict_proba"):
            score = pipe.predict_proba(X_test)[:, 1]
        else:
            score = pipe.decision_function(X_test)

        reporte = classification_report(
            y_test, y_pred, target_names=["Negativo", "Positivo"], digits=4, output_dict=True
        )
        assert isinstance(reporte, dict)

        auc = float(roc_auc_score(y_test, score))

        print(f"\n{'=' * 46}\n  EVALUACIÓN {nombre.upper()}\n{'=' * 46}")
        print(
            classification_report(y_test, y_pred, target_names=["Negativo", "Positivo"], digits=4)
        )
        print(f"ROC-AUC Score: {auc:.4f}")

        resultados.append(
            {
                "Modelo": nombre,
                "Accuracy": reporte["accuracy"],
                "F1 Negativo": reporte["Negativo"]["f1-score"],
                "Recall Negativo": reporte["Negativo"]["recall"],
                "F1 Positivo": reporte["Positivo"]["f1-score"],
                "ROC-AUC": auc,
            }
        )

    return pd.DataFrame(resultados).set_index("Modelo")
