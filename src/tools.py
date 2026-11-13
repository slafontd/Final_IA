from .data_loader import classify_pm25
from .model import predict_next_day

def get_pm25(df, estacion):
    subset = df[df["estacion"].str.lower() == estacion.lower()]
    if subset.empty:
        return {"error": f"No hay registros para la estación {estacion}."}
    last = subset.sort_values("fecha").iloc[-1]
    return {
        "estacion": str(last["estacion"]),
        "fecha": str(last["fecha"]),
        "pm25_ug_m3": float(last["pm25"]),
        "categoria_referencia": classify_pm25(float(last["pm25"]))
    }

def predict_category(model, df, estacion):
    return predict_next_day(model, df, estacion)
