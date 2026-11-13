import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from src.data_loader import load_data
from src.model import train_and_evaluate
from src.rag import SimpleRAG
from src.agent import ask_agent
from src.tools import get_pm25, predict_category

load_dotenv()
st.set_page_config(page_title="AIRE IA", page_icon="🌱", layout="wide")
st.title("🌱 AIRE IA")
st.caption("Asistente experimental de calidad del aire | Datos SISAIRE-AMVA / PM2.5")

try:
    df = load_data()
except Exception as e:
    st.error(f"No se pudo cargar data/pm25.csv: {e}")
    st.stop()

if "es_simulado" in df.columns and df["es_simulado"].astype(str).str.lower().isin(["true","1","si"]).any():
    st.warning("MODO DEMOSTRACIÓN: el CSV contiene datos SIMULADOS.")
else:
    st.success("Datos locales cargados. Verifica la fuente y el periodo en la documentación del proyecto.")

try:
    model, metrics = train_and_evaluate(df)
except Exception as e:
    st.error(f"No se pudo entrenar el modelo temporal: {e}")
    st.stop()

stations = sorted(df["estacion"].unique())
c1,c2,c3 = st.columns(3)
station = c1.selectbox("Estación", stations)
min_date, max_date = df["fecha"].min().date(), df["fecha"].max().date()
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

st.subheader("Modelo clásico: predicción temporal de categoría")
m1,m2,m3 = st.columns(3)
m1.metric("Accuracy modelo", f"{metrics['accuracy_modelo']:.3f}")
m2.metric("Accuracy baseline", f"{metrics['accuracy_baseline']:.3f}")
m3.metric("F1 macro", f"{metrics['f1_macro']:.3f}")
st.caption(f"{metrics['tipo_evaluacion']} | {metrics['n_ejemplos']} ejemplos ({metrics['n_entrenamiento']} entrenamiento / {metrics['n_prueba']} prueba).")
with st.expander("Reporte de clasificación"):
    st.code(metrics["reporte"])

st.subheader("Predicción experimental del próximo día")
if st.button("Predecir categoría del día siguiente"):
    result = predict_category(model, df, station)
    if "error" in result:
        st.error(result["error"])
    else:
        st.info(f"Categoría estimada para {result['fecha_objetivo']}: **{result['categoria_predicha']}**")
        st.caption(result["nota"])

st.subheader("💬 Chat con el agente AIRE IA")
st.caption("Herramientas: consulta de PM2.5 y predicción de categoría del día siguiente. RAG documental con TF-IDF.")
if "chat" not in st.session_state:
    st.session_state.chat = []
for item in st.session_state.chat:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])
prompt = st.chat_input("Ej.: Consulta el último PM2.5 de esta estación o predice la categoría de mañana")
if prompt:
    st.session_state.chat.append({"role":"user","content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Consultando agente..."):
            answer = ask_agent(prompt, df, model, SimpleRAG())
        st.markdown(answer)
    st.session_state.chat.append({"role":"assistant","content":answer})

st.divider()
st.caption("Proyecto académico. El modelo es experimental; no es un sistema oficial de alertas ni sustituye recomendaciones de autoridades sanitarias.")
