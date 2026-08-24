# 📦 Olist Store — Machine Learning Platform

> Sistema modular de Machine Learning para e-commerce (Olist Brasil). Actualmente implementa el pipeline de predicción de tiempos de entrega en días a partir de variables geoespaciales, logísticas y transaccionales.

---

## 🎯 Objetivo de Negocio

Predecir con precisión los **días reales de entrega** (`target_days`) de una orden para optimizar la promesa de entrega al cliente, detectar fricciones logísticas interestatales y reducir costos operativos.

```mermaid
flowchart LR
    A[Datasets Relacionales Olist] --> B[src.common.data]
    B --> C[src.delivery.features]
    C --> D[Modelado TransformedTargetRegressor]
    D --> E[Diagnóstico & Visualizaciones]
```

---

## 🏛️ Arquitectura del Repositorio (*Screaming Architecture*)

La estructura del código está desacoplada por dominios de negocio:

```text
olist_store/
├── datasets/                 # Datos relacionales de Olist (CSVs)
├── notebooks/
│   └── 01_predict_delivery.ipynb  # Pipeline end-to-end de predicción
├── src/
│   ├── common/               # Configuración central y cargadores comunes
│   │   ├── config.py         # Resolución determinista de rutas del proyecto
│   │   └── data.py           # Ingesta de las 6 tablas relacionales
│   ├── delivery/             # Dominio: Tiempos de Entrega (Completado)
│   │   ├── features.py       # Haversine, cubicaje, merges y target
│   │   └── viz.py            # Scatter real vs pred, Permutation Importance, Residuos
│   ├── recommendations/      # Dominio: Recomendación de Productos (Esqueleto)
│   └── sentiment/            # Dominio: Análisis de Reseñas NLP (Esqueleto)
└── pyproject.toml            # Dependencias y configuración de build (uv + hatchling)
```

---

## 🚀 Inicio Rápido

### 1. Requisitos

- Python `>= 3.12`
- [`uv`](https://docs.astral.sh/uv/) (gestor de entornos y paquetes)

### 2. Instalación y Sincronización del Entorno

```bash
# Clona el repositorio y navega al directorio
cd olist_store

# Sincroniza dependencias e instala el paquete editable src
uv sync
```

### 3. Ejecución del Pipeline

Abrí el notebook en Jupyter o VS Code seleccionando el kernel del entorno virtual (`.venv`):

```bash
uv run jupyter lab notebooks/01_predict_delivery.ipynb
```

---

## 🔬 Pipeline de Entrega: Decisiones Técnicas Clave

| Etapa | Decisión Técnica | Justificación de Negocio / Matemática |
| :--- | :--- | :--- |
| **Ingeniería Geoespacial** | **Distancia Haversine** | Cálculo ortodrómico real sobre la curvatura terrestre entre cliente y vendedor. |
| **Consistencia Logística** | **Filtro Mono-vendedor** | Excluye órdenes con múltiples orígenes para evitar ambigüedad en origen y flete. |
| **Validación Temporal** | `shuffle=False` (70/30) | Evita *data leakage* cronológico (entrena con el pasado, prueba en el futuro). |
| **Heterocedasticidad** | $\log(1 + y)$ | `TransformedTargetRegressor` estabiliza la varianza y reduce el impacto de colas largas. |
| **Limpieza de Anomalías** | Recorte al percentil 99 | Elimina disputas y extravíos extremos (> percentil 99) conservando 93.777 órdenes. |

---

## 📊 Resultados y Comparativa de Modelos

Evaluación sobre el conjunto de test (30% final cronológico):

| Modelo | MAE (Días) | RMSE (Días) | $R^2$ | Estado |
| :--- | :---: | :---: | :---: | :---: |
| **Decision Tree Regressor** | 4.31 | 5.86 | 0.119 | *Árbol simple* |
| **Random Forest Regressor** | 3.78 | **5.21** | 0.291 | *Ensemble Bagging* |
| **HistGradientBoostingRegressor** | **3.75** | 5.22 | **0.292** | 🏆 **Ganador (Producción)** |

> 💡 **Variables más influyentes** (según *Permutation Feature Importance*): `distance_km`, `estimated_days`, `total_volume_cm3`, `total_weight_g` y la fricción interestatal (`is_same_state`).

---

## 🛠️ Calidad de Código y Estándares

El proyecto cumple con estándares estrictos de tipado y estilo:

```bash
# Verificación de linters y formato
uv run ruff check src/
uv run ruff format --check src/

# Verificación de tipos estática
uvx pyright src/
```
