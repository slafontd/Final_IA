import json
import os
from openai import OpenAI
from .tools import get_pm25, predict_category

SYSTEM = """Eres AIRE IA, un asistente educativo sobre calidad del aire del Valle de Aburrá.
Responde en español. Usa las herramientas cuando la pregunta requiera datos o predicción.
Usa el contexto documental recuperado para explicar categorías y recomendaciones; cita el título/fuente
del fragmento cuando lo uses. No inventes mediciones. Aclara que el resultado del modelo es experimental,
no un pronóstico oficial ni consejo médico. Si no hay evidencia suficiente, dilo."""

def ask_agent(question, df, model, rag):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return ("No hay GROQ_API_KEY configurada. El dashboard y el modelo clásico funcionan, "
                "pero para demostrar el agente LLM debes configurar una clave de Groq en el archivo .env.")
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    retrieved = rag.retrieve(question)
    context = "\n\n".join(x["text"] for x in retrieved) or "No se recuperaron fragmentos relevantes."
    tools = [
        {"type":"function","function":{"name":"get_pm25","description":"Consulta la última medición disponible en los datos locales para una estación.",
         "parameters":{"type":"object","properties":{"estacion":{"type":"string"}},"required":["estacion"]}}},
        {"type":"function","function":{"name":"predict_category","description":"Clasifica una medición PM2.5 usando el modelo entrenado del proyecto.",
         "parameters":{"type":"object","properties":{"estacion":{"type":"string"},"pm25":{"type":"number"},
          "temperatura":{"type":"number"},"humedad":{"type":"number"},"viento":{"type":"number"}},
          "required":["estacion","pm25"]}}}
    ]
    messages = [
        {"role":"system","content":SYSTEM + "\n\nCONTEXTO RAG:\n" + context},
        {"role":"user","content":question}
    ]
    response = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL","openai/gpt-oss-20b"),
        messages=messages, tools=tools, tool_choice="auto", temperature=0.2
    )
    msg = response.choices[0].message
    if msg.tool_calls:
        messages.append(msg)
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            if call.function.name == "get_pm25":
                result = get_pm25(df, args["estacion"])
            elif call.function.name == "predict_category":
                result = predict_category(model, **args)
            else:
                result = {"error":"Herramienta desconocida"}
            messages.append({"role":"tool","tool_call_id":call.id,"content":json.dumps(result,ensure_ascii=False)})
        response = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL","openai/gpt-oss-20b"),
            messages=messages, temperature=0.2
        )
        return response.choices[0].message.content
    return msg.content or "No se obtuvo respuesta."
