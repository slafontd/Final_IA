import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "pm25_lag0", "pm25_lag1", "pm25_lag2",
    "pm25_media_3", "pm25_media_7", "cambio_1",
    "dia_sin", "dia_cos"
]

def build_supervised_data(df):
    """
    Construye un problema temporal: usar mediciones hasta el día t
    para predecir la categoría del día t+1.
    """
    parts = []
    for station, group in df.sort_values("fecha").groupby("estacion"):
        g = group.sort_values("fecha").copy()
        pm = pd.to_numeric(g["pm25"], errors="coerce")
        g["pm25_lag0"] = pm
        g["pm25_lag1"] = pm.shift(1)
        g["pm25_lag2"] = pm.shift(2)
        g["pm25_media_3"] = pm.rolling(3, min_periods=3).mean()
        g["pm25_media_7"] = pm.rolling(7, min_periods=7).mean()
        g["cambio_1"] = pm - pm.shift(1)
        day = g["fecha"].dt.dayofyear
        g["dia_sin"] = np.sin(2 * np.pi * day / 365.25)
        g["dia_cos"] = np.cos(2 * np.pi * day / 365.25)
        g["objetivo"] = g["categoria"].shift(-1)
        g["fecha_objetivo"] = g["fecha"].shift(-1)
        parts.append(g)
    data = pd.concat(parts, ignore_index=True)
    return data.dropna(subset=FEATURES + ["objetivo"]).sort_values("fecha_objetivo")

def train_and_evaluate(df):
    """Entrena regresión logística con partición temporal y baseline de persistencia."""
    data = build_supervised_data(df)
    if len(data) < 30:
        raise ValueError(
            f"Hay {len(data)} ejemplos temporales válidos. Se requieren al menos 30; "
            "descarga un periodo mayor o agrega más estaciones."
        )
    X = data[FEATURES]
    y = data["objetivo"].astype(str)
    split = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    if y_train.nunique() < 2:
        raise ValueError("El periodo de entrenamiento contiene una sola categoría. Se necesita más variedad temporal.")
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    # Baseline de persistencia: mañana tendrá la categoría observada hoy.
    baseline_pred = data["categoria"].iloc[split:].astype(str).to_numpy()
    metrics = {
        "accuracy_modelo": round(accuracy_score(y_test, pred), 4),
        "accuracy_baseline": round(accuracy_score(y_test, baseline_pred), 4),
        "f1_macro": round(f1_score(y_test, pred, average="macro", zero_division=0), 4),
        "n_ejemplos": int(len(data)),
        "n_entrenamiento": int(len(X_train)),
        "n_prueba": int(len(X_test)),
        "tipo_evaluacion": "Partición temporal 80/20; objetivo: categoría del día siguiente",
        "reporte": classification_report(y_test, pred, zero_division=0)
    }
    return model, metrics

def predict_next_day(model, df, estacion):
    data = build_supervised_data(df)
    station_data = df[df["estacion"].str.lower() == estacion.lower()].sort_values("fecha").copy()
    if len(station_data) < 7:
        return {"error": "Se necesitan al menos 7 mediciones de la estación para construir las variables temporales."}
    pm = pd.to_numeric(station_data["pm25"], errors="coerce")
    last_date = station_data["fecha"].iloc[-1]
    day = last_date.dayofyear
    row = {
        "pm25_lag0": pm.iloc[-1],
        "pm25_lag1": pm.iloc[-2],
        "pm25_lag2": pm.iloc[-3],
        "pm25_media_3": pm.tail(3).mean(),
        "pm25_media_7": pm.tail(7).mean(),
        "cambio_1": pm.iloc[-1] - pm.iloc[-2],
        "dia_sin": np.sin(2 * np.pi * day / 365.25),
        "dia_cos": np.cos(2 * np.pi * day / 365.25)
    }
    prediction = str(model.predict(pd.DataFrame([row])[FEATURES])[0])
    return {
        "estacion": estacion,
        "ultima_fecha_disponible": str(last_date.date()),
        "fecha_objetivo": str((last_date + pd.Timedelta(days=1)).date()),
        "categoria_predicha": prediction,
        "ultima_medicion_pm25": float(pm.iloc[-1]),
        "nota": "Estimación experimental de categoría para el día siguiente; no es pronóstico oficial."
    }
