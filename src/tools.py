
import unicodedata
import pandas as pd

from .data_loader import classify_pm25
from .model import predict_next_day


def normalize_name(value):
    """Normaliza nombres para evitar errores por mayúsculas y tildes."""
    value = str(value).strip().lower()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(
        char for char in value
        if not unicodedata.combining(char)
    )
    return " ".join(value.split())


def find_station(df, station_query):
    """Busca una estación por nombre exacto o coincidencia parcial."""
    if "estacion" not in df.columns:
        return None

    stations = df["estacion"].dropna().unique()
    query = normalize_name(station_query)

    # Primero intenta coincidencia exacta.
    exact = [
        station for station in stations
        if normalize_name(station) == query
    ]

    if len(exact) == 1:
        return exact[0]

    # Después intenta coincidencia parcial.
    matches = [
        station for station in stations
        if query in normalize_name(station)
    ]

    if len(matches) == 1:
        return matches[0]

    # Evita seleccionar arbitrariamente si hay varias estaciones.
    return None


def get_pm25(df, estacion):
    """Consulta la última medición disponible de una estación."""
    station = find_station(df, estacion)

    if station is None:
        disponibles = df["estacion"].dropna().unique().tolist()
        return {
            "error": "No se encontró una coincidencia única.",
            "estaciones_disponibles": disponibles
        }

    subset = df[df["estacion"] == station].copy()
    subset = subset.sort_values("fecha")

    last = subset.iloc[-1]
    value = float(last["pm25"])

    return {
        "estacion": str(station),
        "fecha": str(last["fecha"]),
        "pm25_ug_m3": value,
        "categoria_referencia": classify_pm25(value),
        "total_mediciones": int(len(subset))
    }


def predict_category(model, df, estacion):
    """Predice la categoría del día siguiente."""
    station = find_station(df, estacion)

    if station is None:
        return {
            "error": "No se encontró una coincidencia única para la estación.",
            "estaciones_disponibles": df["estacion"].dropna().unique().tolist()
        }

    # Se envía el nombre canónico exacto al modelo.
    return predict_next_day(model, df, station)
