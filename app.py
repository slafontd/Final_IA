import os
import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from src.data_loader import load_data
from src.model import train_and_evaluate
from src.rag import SimpleRAG
from src.agent import ask_agent
from src.tools import get_pm25

load_dotenv()
st.set_page_config(page_title="AIRE IA", page_icon="🌱", layout="wide")
st.title("🌱 AIRE IA")
st.caption("Asistente experimental de calidad del aire del Valle de Aburrá | SIATA / PM2.5")

try:
    df = load_data()
except Exception as e:
    st.error(f"No se pudo cargar data/pm25.csv: {e}")
    st.stop()

simulated = "es_simulado" in df.columns and df["es_simulado"].astype(str).str.lower().isin(["true","1","si"]).any()
if simulated:
    st.warning("MODO DEMOSTRACIÓN: el CSV incluido contiene datos SIMULADOS. Reemplázalo por mediciones reales antes de entregar.")
else:
    st.success("Datos cargados desde el CSV local. Verifica y documenta su procedencia y fecha de descarga.")

try:
    model, metrics = train_and_evaluate(df)
except Exception as e:
    st.error(f"No se pudo entrenar el modelo: {e}")
    st.stop()

stations = sorted(df["estacion"].unique())
c1,c2,c3 = st.columns(3)
station = c1.selectbox("Estación", stations)
min_date = df["fecha"].min().date()
max_date = df["fecha"].max().date()
date_range = c2.date_input("Periodo", value=(min_date,max_date), min_value=min_date, max_value=max_date)
latest = get_pm25(df, station)
c3.metric("Último PM2.5 (µg/m³)", f"{latest.get('pm25_ug_m3',0):.1f}" if "error" not in latest else "N/D")

filtered = df[df["estacion"] == station].copy()
if isinstance(date_range, tuple) and len(date_range)==2:
    filtered = filtered[(filtered["fecha"].dt.date >= date_range[0]) & (filtered["fecha"].dt.date <= date_range[1])]
left,right = st.columns(2)
with left:
    st.subheader("Evolución de PM2.5")
    st.plotly_chart(px.line(filtered, x="fecha", y="pm25", title=f"PM2.5 — {station}",
                            labels={"fecha":"Fecha","pm25":"PM2.5 (µg/m³)"}), use_container_width=True)
with right:
    st.subheader("Distribución por categoría")
    counts = filtered["categoria"].value_counts().rename_axis("Categoría").reset_index(name="Registros")
    st.plotly_chart(px.bar(counts, x="Categoría", y="Registros", title="Categorías de referencia"),
                    use_container_width=True)

st.subheader("Modelo clásico: Regresión logística")
m1,m2,m3 = st.columns(3)
m1.metric("Accuracy modelo", f"{metrics['accuracy_modelo']:.3f}")
m2.metric("Accuracy baseline", f"{metrics['accuracy_baseline']:.3f}")
m3.metric("F1 macro", f"{metrics['f1_macro']:.3f}")
with st.expander("Reporte de clasificación"):
    st.code(metrics["reporte"])

st.subheader("Consulta y predicción")
p1,p2,p3 = st.columns(3)
pm = p1.number_input("PM2.5 (µg/m³)", min_value=0.0, max_value=600.0, value=float(latest.get("pm25_ug_m3",20.0)))
temp = p2.number_input("Temperatura (°C)", value=24.0)
hum = p3.number_input("Humedad (%)", min_value=0.0, max_value=100.0, value=65.0)
if st.button("Clasificar medición"):
    from src.tools import predict_category
    result = predict_category(model, station, pm, temp, hum)
    st.info(f"Categoría estimada por el modelo: **{result['categoria_modelo']}**")

st.subheader("💬 Chat con el agente AIRE IA")
st.caption("El agente dispone de dos herramientas: consulta de PM2.5 y clasificación con el modelo. También recupera contexto del corpus documental.")
if "chat" not in st.session_state:
    st.session_state.chat = []
for item in st.session_state.chat:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])
prompt = st.chat_input("Ej.: Consulta PM2.5 de Tráfico Centro o explica qué significa PM2.5")
if prompt:
    st.session_state.chat.append({"role":"user","content":prompt})
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Consultando agente..."):
            answer = ask_agent(prompt, df, model, SimpleRAG())
        st.markdown(answer)
    st.session_state.chat.append({"role":"assistant","content":answer})

st.divider()
st.caption("Proyecto académico. No es un sistema oficial de alertas ni sustituye recomendaciones de autoridades sanitarias.")
