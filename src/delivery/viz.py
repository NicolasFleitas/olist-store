import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance


def plot_real_vs_predicted(
    y_test: pd.Series,
    y_pred,
    title: str = "Valores Reales vs Predicciones (Días de Entrega)",
):
    """Gráfico de dispersión de valores reales vs predichos con línea diagonal ideal."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.3)
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "--r",
        linewidth=2,
    )
    plt.title(title, fontsize=14)
    plt.xlabel("Días Reales", fontsize=14)
    plt.ylabel("Días Predichos", fontsize=14)
    plt.show()


def plot_permutation_importance(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    features: list[str],
) -> pd.DataFrame:
    """Calcula y grafica la importancia de variables por permutación (aumento de RMSE)."""
    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        n_jobs=-1,
        random_state=42,
        scoring="neg_root_mean_squared_error",
    )
    if isinstance(perm, dict) and "importances_mean" not in perm:
        # Multi-metric scoring returns dict[str, Bunch]
        perm_bunch = next(iter(perm.values()))
    else:
        perm_bunch = perm

    imp = pd.DataFrame(
        {
            "Importancia": perm_bunch["importances_mean"],
            "Desvio": perm_bunch["importances_std"],
        },
        index=features,
    ).sort_values("Importancia")

    print(imp.round(3))

    plt.figure(figsize=(10, 6))
    imp["Importancia"].plot.barh(xerr=imp["Desvio"])
    plt.title("Permutation Importance - aumento del RMSE al permutar cada variable")
    plt.xlabel("Aumento del RMSE (dias)")
    plt.tight_layout()
    plt.show()
    return imp


def plot_residuals(y_test: pd.Series, y_pred):
    """Gráfico de residuos (real - predicho) vs valores predichos."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=y_test - y_pred, alpha=0.3)
    plt.axhline(0, color="red", linestyle="--", linewidth=2)
    plt.title("Residuos vs Predicción (Días de Entrega)")
    plt.xlabel("Días Predichos")
    plt.ylabel("Residuo (real - predicho)")
    plt.show()
