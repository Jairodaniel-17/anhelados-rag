import requests
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel
import os
from langchain import hub
from pydantic import BaseModel
from langchain.agents import AgentExecutor, create_react_agent, tool

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/distiluse-base-multilingual-cased",
    encode_kwargs={"normalize_embeddings": True},
)
print("\033c")


@tool
def check_system_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Devuelve la hora actual del sistema en el formato especificado.

    Parámetros:
    - format (str): El formato de la hora a devolver. Por defecto es "%Y-%m-%d %H:%M:%S", puedes cambiarlo según tus necesidades.

    Retorna:
    - str: La hora actual del sistema en el formato especificado.
    """
    from datetime import datetime

    current_time = datetime.now().strftime(format)
    formatted_time = datetime.strptime(current_time, format)
    return formatted_time


def cargar_vector_store(store_name="informacion_de_la_empresa") -> FAISS:
    return FAISS.load_local(
        folder_path=store_name,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )


vector_store = cargar_vector_store("informacion_de_la_empresa")


class Documento(BaseModel):
    contenido: str
    fuente: str
    puntaje: float


def format_output(resultados: list) -> list[Documento]:
    documentos_formateados = []
    for documento, score in resultados:
        doc = Documento(
            contenido=documento.page_content,
            fuente=documento.metadata["source"],
            puntaje=score,
        )
        documentos_formateados.append(doc)
    return documentos_formateados


@tool
def search_context_data_base(query: str, k: int = 4) -> str:
    """
    Busca en la estructura de datos de la base de datos SQLite.
    """
    resultados = vector_store._similarity_search_with_relevance_scores(
        query=query, k=k, filter={"source": "documents\\estructura_base_datos.txt"}
    )
    if len(resultados) == 0:
        return "No se encontraron resultados."

    result = format_output(resultados)
    output = ""
    for i in range(min(len(result), k)):
        output += result[i].contenido

    return output


class ConsultaAPI(BaseModel):
    query: str


@tool
def consultar_db_via_api(query: str):
    """
    Consulta la DB SQLite con una consulta puntual. Máximo puedes solicitar hasta 20 registros.
    NO USES COMILLAS DOBLES AL INICIO Y AL FINAL DE LA CONSULTA.

    Parámetros:
    - query (str): La consulta SQL a ejecutar en la base de datos.

    Retorna:
    - dict: Los resultados de la consulta en formato JSON.
    """
    try:
        # Eliminar comillas dobles al inicio y al final de la consulta
        query = query.strip('"')
        # Eliminar punto y coma al final de la consulta si existe
        if query.endswith(";"):
            query = query[:-1]
        # Agregar barra invertida antes de cada comilla simple
        query = query.replace("'", "\\'")

        print(query)
        print("Consultando la API...")

        format_query_json = {"query": query}
        response = requests.post(
            url="https://jairodanielmt-anhelados.hf.space/execute",
            json=format_query_json,  # Enviar el cuerpo de la solicitud como JSON
            headers={
                "Content-Type": "application/json"
            },  # Asegurar el tipo de contenido
        )
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error al consultar la API: {e}")
        if e.response is not None:
            print(e.response.text)
        return None


prompt = hub.pull("hwchase17/react")
tools = [check_system_time, search_context_data_base, consultar_db_via_api]

llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    temperature=0.3,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=20,
)


def ask_agent(consulta) -> str:
    d = "Eres un asistente, tienes acceso a herramientas tools, antes de ejecutar un comando SQL consulta el contexto de la base de datos hay productos, empleados y más tablas, antes de proceder a generar un comando SQL consulta si tienes todo el contexto de la BD, la siguiente consulta es:"
    query = f"{d} {consulta}"
    output = agent_executor.invoke({"input": query})
    return output["output"]


import streamlit as st

st.title("Chatbot de Anhelados")

if "history" not in st.session_state:
    st.session_state["history"] = []

pregunta = st.chat_input("Escribe tu consulta...")

if pregunta:
    st.session_state["history"].append({"role": "user", "content": pregunta})
    respuesta = ask_agent(pregunta)
    st.session_state["history"].append({"role": "ai", "content": respuesta})

for message in st.session_state["history"]:
    if message["role"] == "user":
        with st.chat_message(name="user", avatar="👩‍💻"):
            st.write(message["content"])
    else:
        with st.chat_message(name="ai", avatar="🍦"):
            st.write(message["content"])
