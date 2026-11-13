# AIRE IA – Asistente Inteligente de Calidad del Aire

## 1. Descripción del proyecto

AIRE IA es un asistente inteligente desarrollado como proyecto académico para el análisis de la calidad del aire en el Valle de Aburrá, con énfasis en la concentración de material particulado fino PM2.5.

El sistema integra técnicas de aprendizaje automático (Machine Learning), modelos de lenguaje de gran tamaño (LLM) y recuperación aumentada por generación (RAG), con el propósito de consultar mediciones de calidad del aire, analizar su comportamiento y generar estimaciones experimentales de categorías de contaminación.

La aplicación cuenta con un dashboard interactivo desarrollado en Streamlit y un agente conversacional que utiliza herramientas especializadas para consultar datos y ejecutar predicciones.

## 2. Objetivos

- Analizar mediciones de PM2.5 provenientes de fuentes de datos de calidad del aire.
- Implementar un modelo de clasificación mediante regresión logística.
- Comparar el desempeño del modelo con un modelo de referencia (baseline).
- Visualizar el comportamiento de las concentraciones de PM2.5 mediante gráficos interactivos.
- Implementar un agente inteligente con capacidad de consultar mediciones y ejecutar predicciones.
- Integrar un sistema RAG para recuperar información documental relacionada con la calidad del aire.

## 3. Tecnologías utilizadas

| Tecnología | Aplicación |
|---|---|
| Python | Lenguaje principal |
| Streamlit | Interfaz web y dashboard |
| Pandas y NumPy | Procesamiento de datos |
| Scikit-learn | Entrenamiento y evaluación de modelos |
| Plotly | Visualizaciones interactivas |
| Groq API | Acceso al modelo de lenguaje |
| OpenAI Python SDK | Comunicación con la API compatible de Groq |
| TF-IDF | Recuperación de información documental |
| Git y GitHub | Control de versiones |

## 4. Fuente de datos

El proyecto utiliza mediciones de PM2.5 provenientes de SISAIRE-AMVA, correspondientes a la estación:

**Parque Biblioteca Fernando Botero – Medellín, San Cristóbal.**

Características del conjunto de datos:

- Variable: PM2.5.
- Unidad: µg/m³.
- Frecuencia: mediciones diarias.
- Periodo: 1 de enero de 2025 al 28 de febrero de 2026.
- Registros: 415.
- Número de estaciones: 1.

Los datos fueron exportados desde la plataforma de consulta de SISAIRE-AMVA y posteriormente procesados para adaptarlos al formato utilizado por la aplicación.

Las variables meteorológicas de temperatura, humedad y viento no están disponibles en el archivo utilizado. Por esta razón, se mantienen vacías y no se generan valores artificiales para ellas.

El proyecto conserva un archivo de datos simulados independiente, utilizado únicamente como respaldo para pruebas de demostración.

## 5. Componentes del sistema

### 5.1. Dashboard interactivo

Desarrollado con Streamlit, permite:

- Consultar las estaciones disponibles.
- Visualizar la evolución temporal de PM2.5.
- Analizar la distribución de mediciones por categorías.
- Consultar la última medición disponible.
- Visualizar las métricas de evaluación del modelo.
- Ejecutar predicciones experimentales.

### 5.2. Modelo de Machine Learning

Se implementa un modelo de regresión logística mediante Scikit-learn.

El modelo utiliza variables temporales construidas a partir del historial de PM2.5, entre ellas:

- Mediciones anteriores.
- Promedios móviles de tres y siete días.
- Variaciones entre mediciones consecutivas.
- Variables temporales derivadas de la fecha.

El objetivo es estimar la categoría de PM2.5 correspondiente al día siguiente.

La evaluación utiliza una partición temporal de los datos, con 80 % para entrenamiento y 20 % para prueba.

Se utiliza un modelo baseline de persistencia como referencia comparativa.

Resultados obtenidos en la ejecución del proyecto:

| Métrica | Resultado |
|---|---:|
| Accuracy del modelo | 0.915 |
| Accuracy del baseline | 0.854 |
| F1 macro | 0.478 |
| Ejemplos temporales | 408 |
| Entrenamiento | 326 |
| Prueba | 82 |

Los resultados corresponden al conjunto de datos utilizado y no representan una garantía de desempeño en otras estaciones o periodos.

### 5.3. Agente inteligente LLM

El agente conversacional utiliza la API de Groq para procesar preguntas en lenguaje natural.

Modelo configurado: `openai/gpt-oss-20b`.

El agente puede interpretar preguntas y utilizar herramientas especializadas para consultar mediciones y ejecutar predicciones.

### 5.4. Herramientas del agente

Se implementan dos herramientas principales:

**get_pm25**

Consulta la última medición disponible de PM2.5 para una estación determinada y devuelve su valor, fecha y categoría de referencia.

**predict_category**

Ejecuta el modelo de Machine Learning para estimar la categoría de PM2.5 del día siguiente utilizando el historial disponible de la estación.

### 5.5. Sistema RAG

El sistema de recuperación aumentada por generación utiliza TF-IDF para recuperar fragmentos relevantes de un corpus documental sobre calidad del aire.

Este componente permite proporcionar contexto documental al agente para responder preguntas relacionadas con:

- Definición del material particulado PM2.5.
- Principales fuentes de contaminación.
- Efectos sobre la salud.
- Clasificación de la calidad del aire.

## 6. Estructura del proyecto

```text
AIRE-IA/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
│
├── data/
│   ├── pm25.csv
│   └── pm25_demo.csv
│
├── docs/
│   └── corpus_calidad_aire.md
│
├── evaluation/
│   └── preguntas.json
│
└── src/
    ├── __init__.py
    ├── data_loader.py
    ├── model.py
    ├── rag.py
    ├── agent.py
    └── tools.py
```

## 7. Instalación y configuración

### Requisitos

- Python 3.10 o superior.
- Visual Studio Code.
- Conexión a internet.
- Una cuenta en Groq para habilitar el agente LLM.

### Clonar el repositorio

```bash
git clone https://github.com/slafontd/Final_IA.git
cd Final_IA
```

### Crear el entorno virtual

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Instalar dependencias

```powershell
pip install -r requirements.txt
```

### Configurar las variables de entorno

Crear el archivo `.env` a partir de `.env.example` y configurar las siguientes variables:

```env
GROQ_API_KEY=tu_clave_api
GROQ_MODEL=openai/gpt-oss-20b
```

La clave API es privada y no debe publicarse en GitHub.

### Ejecutar la aplicación

```powershell
streamlit run app.py
```

La aplicación estará disponible en el navegador mediante la dirección local indicada por Streamlit.

## 8. Evaluación del agente

El agente se prueba mediante preguntas orientadas a verificar sus capacidades principales:

1. Consulta de la última medición disponible de PM2.5 para una estación.
2. Predicción experimental de la categoría del día siguiente.
3. Recuperación de información documental sobre definición, fuentes y efectos del PM2.5.

Estas pruebas permiten verificar la integración del modelo LLM, las herramientas y el sistema RAG.

## 9. Limitaciones

- El conjunto de datos utilizado corresponde a una sola estación de monitoreo.
- Las mediciones disponibles son diarias y cubren un periodo específico.
- No se dispone de variables meteorológicas en el archivo utilizado.
- El modelo realiza una estimación experimental de categorías y no constituye un sistema oficial de pronóstico.
- El desempeño del modelo presenta diferencias entre métricas, por lo que se requiere especial atención al F1 macro.
- Las categorías utilizadas son referencias simplificadas para el análisis académico.
- El agente depende de la disponibilidad del servicio externo de Groq y de una API Key válida.

## 10. Consideraciones de uso

AIRE IA es un prototipo académico y experimental. Sus resultados no sustituyen los sistemas oficiales de monitoreo, las alertas emitidas por las autoridades ambientales ni las recomendaciones de profesionales de la salud.

Para consultar información actualizada sobre la calidad del aire se recomienda acudir a las fuentes oficiales de monitoreo.

## 11. Autores
Santiago Lafont Díaz

Proyecto académico desarrollado individualmente.

**Repositorio:** https://github.com/slafontd/Final_IA

