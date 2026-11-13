from pathlib import Path
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "pm25.csv"

def load_data(path=DATA_PATH):
    """Carga el CSV en formato canónico: fecha, estacion, pm25, temperatura, humedad, viento."""
    df = pd.read_csv(path)
    required = {"fecha", "estacion", "pm25"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias en pm25.csv: {', '.join(sorted(missing))}")
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    for col in ["pm25", "temperatura", "humedad", "viento"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["fecha", "estacion", "pm25"]).copy()
    df["estacion"] = df["estacion"].astype(str)
    df["categoria"] = df["pm25"].apply(classify_pm25)
    return df.sort_values("fecha")

def classify_pm25(value):
    """Categorías simplificadas de referencia para PM2.5 (µg/m³), basadas en tabla institucional."""
    if value <= 12:
        return "Buena"
    if value <= 37:
        return "Aceptable"
    if value <= 55:
        return "Dañina para grupos sensibles"
    if value <= 150:
        return "Dañina"
    if value <= 250:
        return "Muy dañina"
    return "Peligrosa"
