# RAGent

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/TheRamdomX/RAGent)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-00B86B?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyBmaWxsPSIjMDBiODZiIiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgaGVpZ2h0PSIxNiI+PHBhdGggZD0iTTUgMTMuNXYtMy41aDEwVjEzaC0xMHptMCA1di0zLjVoMTB2My41aC0xMHptMTAtMTBoLTl2LTNoOSB2M3oiLz48L3N2Zz4=)](https://www.langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-FFD700?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyBmaWxsPSIjRkZEMzAwIiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgaGVpZ2h0PSIxNiI+PHBhdGggZD0iTTExLjUgM2MuODMgMCAxLjUgLjY3IDEuNSAxLjUgMCAuODMtLjY3IDEuNS0xLjUgMS41LS44MyAwLTEuNS0uNjctMS41LTEuNSAwLS44My42Ny0xLjUgMS41LTEuNXpNNSA5Yy44MyAwIDEuNS42NyAxLjUgMS41IDAgLjgzLS42NyAxLjUtMS41IDEuNS0uODMgMC0xLjUtLjY3LTEuNS0xLjUgMC0uODMuNjctMS41IDEuNS0xLjV6bTAgNWMuODMgMCAxLjUuNjcgMS41IDEuNSAwIC44My0uNjcgMS41LTEuNSAxLjUtLjgzIDAtMS41LS42Ny0xLjUtMS41IDAtLjgzLjY3LTEuNSAxLjUtMS41ek0xOSA5Yy44MyAwIDEuNS42NyAxLjUgMS41IDAgLjgzLS42NyAxLjUtMS41IDEuNS0uODMgMC0xLjUtLjY3LTEuNS0xLjUgMC0uODMuNjctMS41IDEuNS0xLjV6bTAgNWMuODMgMCAxLjUuNjcgMS41IDEuNSAwIC44My0uNjcgMS41LTEuNSAxLjUtLjgzIDAtMS41LS42Ny0xLjUtMS41IDAtLjgzLjY3LTEuNSAxLjUtMS41eiIvPjwvc3ZnPg==)](https://www.trychroma.com/)
[![PyPDF2](https://img.shields.io/badge/PyPDF2-008080?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)](https://pypi.org/project/PyPDF2/)
[![python-docx](https://img.shields.io/badge/python--docx-0078D4?style=for-the-badge&logo=microsoftword&logoColor=white)](https://pypi.org/project/python-docx/)
[![tiktoken](https://img.shields.io/badge/tiktoken-FF4500?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyBmaWxsPSIjRkY0NTAwIiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgaGVpZ2h0PSIxNiI+PHBhdGggZD0iTTUgMTMuNXYtMy41aDEwVjEzaC0xMHptMCA1di0zLjVoMTB2My41aC0xMHptMTAtMTBoLTl2LTNoOSB2M3oiLz48L3N2Zz4=)](https://github.com/openai/tiktoken)
[![marker-pdf](https://img.shields.io/badge/marker--pdf-FFB300?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)](https://pypi.org/project/marker-pdf/)
[![dotenv](https://img.shields.io/badge/dotenv-10AA50?style=for-the-badge&logo=dotenv&logoColor=white)](https://pypi.org/project/python-dotenv/)

RAGent es un asistente conversacional basado en RAG (Retrieval-Augmented Generation) que responde preguntas utilizando información extraída de documentos (PDF, DOCX, TXT) y modelos de lenguaje (LLM). El sistema ingesta archivos, los procesa en chunks, genera embeddings, almacena los vectores en una base ChromaDB y utiliza un modelo LLM para responder preguntas apoyándose en el contexto recuperado.

Además, expone:

- API HTTP con FastAPI (`api.py`) para consultas desde un frontend u otros clientes.
- CLI con Typer (`main.py`) para ingestar, chatear por consola, listar y eliminar documentos.


## Motivacion

Otorgarle al estudiante la capacidad de poder estudiar en la misma plataforma donde se encuentra el material de semestres anteriores y no requerir de herramientas ajenas a la universidad. Esto permite generar un ecosistema de aprendizaje unificado, donde el alumno no pierde tiempo saltando entre aplicaciones, evitando la dependencia de plataformas externas.
Además, centralizar los recursos académicos abre la puerta a experiencias de estudio más inteligentes, como búsqueda semántica, generación de resúmenes, creación automática de flashcards y mapas mentales basados en el material oficial. Esto beneficia tanto a quienes buscan repasar conceptos clave como a quienes necesitan apoyo para organizar su propio proceso de estudio.
En conjunto, esto se traduce en una plataforma más eficiente, accesible y personalizada, alineada con la necesidad actual de herramientas que potencien el rendimiento académico sin fricciones ni barreras tecnológicas.

## Tecnologías Utilizadas


### Lenguaje & Runtime
- **Python** 3.8+ - Lenguaje de programación principal

### Core Framework & LLM
- **[LangChain](https://www.langchain.com/)** (>=0.1.0) - Framework principal para construcción de aplicaciones con LLMs
- **[OpenAI API](https://openai.com/)** (>=1.0.0) - API de OpenAI para modelos GPT y embeddings
- **[LangChain OpenAI](https://python.langchain.com/docs/integrations/platforms/openai)** (>=0.3.33) - Integración con modelos OpenAI
- **[LangChain Community](https://python.langchain.com/docs/integrations/platforms/)** (>=0.0.100) - Integraciones adicionales de LangChain

### Vector Database & RAG
- **[ChromaDB](https://www.trychroma.com/)** (>=0.3.24) - Base de datos vectorial para almacenamiento de embeddings
- **[LangChain Chroma](https://python.langchain.com/docs/integrations/vectorstores/chroma)** (>=0.2.6) - Integración de ChromaDB con LangChain

### Procesamiento de Documentos
- **[PyPDF2](https://pypi.org/project/PyPDF2/)** (>=3.0.0) - Extracción de texto de archivos PDF
- **[python-docx](https://pypi.org/project/python-docx/)** (>=0.8.11) - Procesamiento de documentos Word (.docx)
- **[marker-pdf](https://pypi.org/project/marker-pdf/)** (>=1.0.0) - OCR avanzado y parsing de PDFs con contenido complejo
- **[tiktoken](https://github.com/openai/tiktoken)** (>=0.11.0) - Tokenizador de OpenAI para gestión de límites de tokens

### API & Web Framework
- **[FastAPI](https://fastapi.tiangolo.com/)** (>=0.104.0) - Construcción de API
- **[Uvicorn](https://www.uvicorn.org/)** (>=0.24.0) - Servidor ASGI para FastAPI
- **[Pydantic](https://docs.pydantic.dev/)** (>=2.0.0) - Validación de datos y serialización



### Arquitectura de Agentes
- **ReAct Agent** - Implementación del patrón Reasoning + Acting para consultas complejas
- **RAG (Retrieval-Augmented Generation)** - Arquitectura para respuestas basadas en documentos recuperados


## Flujo de datos

### Ingesta de archivos

1. El usuario ingresa archivos mediante la CLI ([main.py](main.py), [ingestion.py](app/data/ingestion.py)).

2. Los archivos se leen y procesan (PDF, DOCX, TXT).

3. El texto se divide en chunks ([chunking.py](app/data/chunking.py)). Nota: Por el poco tiempo, se genera un único chunk grande por archivo, preservando el contexto completo del documento.

4. Se generan embeddings para cada chunk ([embeddings.py](app/models/embeddings.py)).

5. Los chunks y sus embeddings se almacenan en ChromaDB.

### Recuperación y respuesta

1. El usuario realiza una consulta en el chat.

2. El sistema recupera los chunks más relevantes desde ChromaDB usando embeddings ([retriever.py](app/rag/retriever.py)).

3. Un agente ReAct (`app/rag/ReAct.py`) orquesta el proceso, llama a la herramienta `RAG_Search` que otorga contexto adicional al promt.

4. Se construye un prompt con el contexto recuperado y la pregunta del usuario ([qa.py](app/rag/qa.py)).

5. El prompt se envía al modelo LLM para generar una respuesta ([llm.py](app/models/llm.py)).

6. Se muestra la respuesta y las fuentes relevantes al usuario.




## Componentes principales

- [main.py](main.py): CLI para ingesta y ejecución.
- [app/data/ingestion.py](app/data/ingestion.py): Procesamiento de archivos y chunks.
- [app/models/embeddings.py](app/models/embeddings.py): Generación de embeddings.
- [app/rag/retriever.py](app/rag/retriever.py): Recuperación de contexto relevante.
- [app/rag/qa.py](app/rag/qa.py): Construcción de prompts y respuestas.
- [app/rag/ReAct.py](app/rag/ReAct.py): Agente ReAct que orquesta razonamiento y llamadas a la herramienta RAG.
- [app/models/llm.py](app/models/llm.py): Interfaz con el modelo LLM.

## API HTTP

Servidor FastAPI desde `api.py`.

- POST `/api/query` — Consulta al chatbot.
    - Request JSON:
        - `prompt` (str): pregunta del usuario.
        - `ramo` (str): nombre de la colección/ramo (p. ej., `study_collection`).
        - `use_rag` (bool, opcional): override para usar/no usar RAG (por defecto true).
    - Response JSON:
        - `answer` (str): respuesta del asistente.
        - `sources` (string[], opcional): nombres de archivos fuente deduplicados, si hubo contexto.
        - `ramo` (str): colección usada.

Ejemplo (cURL opcional):

```bash
curl -X POST http://localhost:8000/api/query \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"¿Qué es una derivada?","ramo":"study_collection","use_rag":true}'
```

## Uso básico (CLI)

- Ingesta de un archivo (colección opcional, por defecto `study_collection`):

```bash
python main.py ingest docs/algebra.pdf study_collection
```

- Ingesta de múltiples archivos con `--paths` (colección al final):

```bash
python main.py ingest --paths docs/a.pdf docs/notes.docx study_collection
```

- Modo chat (por defecto usa RAG y `study_collection`):

```bash
python main.py chat
```

- Ingesta + chat en un solo paso (posicionales `paths`):

```bash
python main.py run docs/file.pdf
```

- Listar archivos del RAG (usar `-a` para ver ids de ejemplo):

```bash
python main.py list
python main.py list -a
```

- Eliminar por archivo(s) o ids:

```bash
python main.py delete files/doc.pdf other.pdf
python main.py delete --ids id1,id2,id3
```

## Configuración (variables de entorno)

RAGent se configura principalmente mediante variables de entorno (usando `.env`). Las más relevantes:

- `OPENAI_API_KEY` — Clave de OpenAI para embeddings y LLM.
- `CHROMA_PERSIST_DIR` — Directorio donde se persiste la base ChromaDB (por defecto `./chroma_db`).
- `EMBEDDING_MODEL` — Modelo de embeddings (por defecto `text-embedding-3-small`).
- `LLM_MODEL` — Modelo LLM para generación (por defecto `gpt-4.1-nano`).
- `DEFAULT_TOP_K` — Número por defecto de documentos a recuperar en búsquedas.

- OCR y extracción de PDFs:
    - Ahora se usa siempre Marker OCR para PDFs durante la ingesta. Si fallara, se hace fallback a PyPDF2 automáticamente.
    - Variables relacionadas (si las usabas antes): `FORCE_MARKER_OCR`, `MARKER_OCR_THRESHOLD` (ya no afectan el comportamiento por defecto).

- Chunking y deduplicación:
    - `CHUNK_DEFAULT_SIZE` — Tamaño por defecto del chunk en caracteres (por defecto 400000, se usa 1 chunk por archivo).
    - `CHUNK_DEFAULT_OVERLAP` — Solapamiento entre chunks en caracteres (por defecto 100000; irrelevante con 1 chunk).
    - `MIN_CHUNK_CHARS` — Longitud mínima aceptable para un chunk (por defecto 500).
    - `DEDUP_SIM_THRESHOLD` — Umbral de similitud para marcar near-duplicates usando embeddings (0..1, por defecto 0.9).

- LLM / Agentes:
    - `LLM_TEMPERATURE` — Controla creatividad/determinismo del LLM (0.0 a 1.0, por defecto 0.7).
    - `LLM_MAX_COMPLETION_TOKENS` — Tokens máximos por respuesta del LLM.
    - `BUDGET_CALLS_PER_QUERY` — Límite de llamadas a herramientas por consulta en agentes (por defecto 5).
    - `RERANK_ENABLED` — Activa re-ranking por similitud coseno (por defecto true).
    - `RERANK_TOP_K` — Candidatos a recuperar antes de re-rankear (por defecto 20).
    - `MAX_MODEL_TOKENS` — Límite aproximado de tokens del modelo (por defecto 300000).
    - `RESERVED_RESPONSE_TOKENS` — Tokens reservados para la respuesta (por defecto 2048).

## Notas sobre OCR

Durante la ingesta de PDFs se invoca Marker OCR de forma incondicional. Si Marker falla, se registra el error y se intenta extraer texto con PyPDF2 como respaldo. Esto mejora la robustez para PDFs escaneados o con poca extracción nativa .

