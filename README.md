# AIRE IA — MVP académico

Asistente experimental de calidad del aire del Valle de Aburrá. Integra clasificación de PM2.5 con regresión logística, comparación con baseline, dashboard Streamlit, RAG documental y agente LLM con dos herramientas.

## 1. Requisitos
- Python 3.10 o superior
- VS Code
- Conexión a internet para instalar paquetes y, si se usa, llamar al LLM

## 2. Instalación (terminal de VS Code)
```bash
python -m venv .venv
```
Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```
macOS/Linux:
```bash
source .venv/bin/activate
```
Luego:
```bash
pip install -r requirements.txt
```

## 3. Ejecutar
```bash
streamlit run app.py
```
La app abre en el navegador.

## 4. Datos
El archivo `data/pm25.csv` incluido es SIMULADO y sirve únicamente para verificar que el código corre. No presentarlo como datos reales. Para cumplir el requisito del curso, descargar el JSON PM2.5 desde el Portal de Datos Abiertos del Área Metropolitana (dataset Mediciones estaciones calidad del aire, publicador SIATA) y convertirlo al CSV canónico con columnas:
`fecha,estacion,pm25,temperatura,humedad,viento`
`temperatura`, `humedad` y `viento` pueden quedar vacías si el dataset PM2.5 no las contiene. En ese caso, el modelo imputará valores; documenta esta limitación. No inventes variables meteorológicas.
Portal: https://datosabiertos.metropol.gov.co/node/99
Distribución PM2.5: https://datosabiertos.metropol.gov.co/sites/default/files/uploaded_resources/Datos_SIATA_Aire_pm25.json

El archivo CSV final debe tener mediciones reales verificables. Conserva fecha de descarga, fuente y transformaciones en el informe. Si el JSON tiene una estructura distinta, inspecciona sus claves antes de mapearlas.

## 5. Configurar agente LLM (Groq)
1. Crea una cuenta en https://console.groq.com/
2. Genera una API key.
3. Copia `.env.example` como `.env`.
4. Pega la clave en `GROQ_API_KEY=...`.
5. Reinicia Streamlit.
No subas `.env` a GitHub. Si no configuras la clave, el dashboard y el modelo clásico funcionan, pero el agente LLM no estará habilitado.

## 6. Componentes
- Modelo: regresión logística propia con scikit-learn; baseline DummyClassifier.
- RAG: recuperación TF-IDF sobre `docs/corpus_calidad_aire.md`.
- Herramientas: `get_pm25` y `predict_category`.
- Interfaz: Streamlit y Plotly.

## 7. Preguntas para evaluación
Usa `evaluation/preguntas.json`. Registra pregunta, herramienta esperada, herramienta observada y si la respuesta citó contexto documental.

## 8. Advertencias
La clasificación por umbrales es una referencia simplificada y el modelo es académico. No es una predicción oficial ni una herramienta médica o de alertas. Los datos simulados deben sustituirse por datos reales antes de la entrega.
