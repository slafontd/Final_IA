from .data_loader import classify_pm25
from .model import predict_one

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

def predict_category(model, estacion, pm25, temperatura=24, humedad=65, viento=1.5):
    row = {"estacion": estacion, "pm25": float(pm25), "temperatura": float(temperatura),
           "humedad": float(humedad), "viento": float(viento)}
    return {"categoria_modelo": predict_one(model, row), "entrada": row}
