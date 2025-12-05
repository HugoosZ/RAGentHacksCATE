from app.rag.retriever import get_relevant_docs, get_vectorstore
from app.models.llm import Agent
from app.utils import config
from app.utils.logger import logger
from typing import Dict, Any, Tuple, List, Optional
import tiktoken

llm = Agent()

SYSTEM_INSTRUCTIONS = (
    "Eres un asistente académico que SOLO puede usar el contexto recuperado desde PDFs. "
    "Si la información no está en el contexto, responde exactamente: 'No disponible en el contexto'. No inventes. "
    "Cita siempre las fuentes como [PDF: <nombre>, pág <n>]. Español claro y directo. "
    "Devuelve SIEMPRE un JSON válido y NADA más (sin texto fuera del objeto JSON). "
    "Modos soportados (input 'mode'): 'qa', 'search', 'flashcards', 'mindmap'. "
    "Esquemas de salida obligatorios por modo: "
    "qa -> {\"answer\":\"string (1–4 oraciones con citas)\","
    "\"explanation_steps\":[\"string\"],"
    "\"citations\":[{\"pdf\":\"string\",\"page\":0,\"snippet\":\"string\"}],"
    "\"sources_used\":[\"string\"],"
    "\"limitations\":\"string o vacío\","
    "\"followups\":[\"string\"]} "
    "search -> {\"query_rewrite\":\"string\","
    "\"top_passages\":[{\"pdf\":\"string\",\"page\":0,\"score\":0.0,\"snippet\":\"string\",\"citation\":\"[PDF: nombre, pág n]\"}],"
    "\"related_queries\":[\"string\"],"
    "\"limitations\":\"string o vacío\"} "
    "flashcards -> {\"cards\":[{\"front\":\"string\",\"back\":\"string con 1–2 citas al final [PDF: nombre, pág n]\",\"tag\":\"string\",\"difficulty\":2}],"
    "\"sources_used\":[\"string\"],"
    "\"limitations\":\"string o vacío\"} "
    "mindmap -> {\"root\":{\"id\":\"root\",\"title\":\"string\",\"notes\":\"2–3 frases con citas [PDF: nombre, pág n]\"},"
    "\"nodes\":[{\"id\":\"n1\",\"title\":\"string\",\"notes\":\"string con citas si aplica\"}],"
    "\"edges\":[{\"from\":\"root\",\"to\":\"n1\",\"label\":\"incluye\"}],"
    "\"citations\":[{\"pdf\":\"string\",\"page\":0}],"
    "\"limitations\":\"string o vacío\"} "
    "Reglas adicionales: usa solo términos presentes en los PDFs; si hay poca evidencia o ambigüedad, rellena 'limitations' y sugiere 'followups'. "
)

def _format_chunk_header(meta: dict, fallback_index: int) -> str:
    src = meta.get("source", "desconocido")
    page = meta.get("page")
    page_str = f", pág {page}" if page is not None else ""
    chunk = meta.get("chunk", fallback_index)
    # Cabecera que ya trae la forma de cita para ayudar al modelo
    return f"[PDF: {src}{page_str}] (chunk={chunk})\n"

def build_prompt(
    context_docs: List[Any],
    question: str,
    mode: str = "qa",
    files_focus: Optional[List[str]] = None,
) -> str:
    """
    Construye un prompt que:
    - Inyecta SYSTEM_INSTRUCTIONS
    - Empaqueta el contexto troceado con cabeceras tipo cita
    - Exige salida SOLO JSON según 'mode'
    """
    max_model_tokens = config.MAX_MODEL_TOKENS
    reserved = config.RESERVED_RESPONSE_TOKENS

    encoding = tiktoken.encoding_for_model(config.LLM_MODEL)

    # Bloque de formato: fuerza salida JSON estricta según mode.
    formatting = (
        "SALIDA ESTRICTA:\n"
        f"- Devuelve SOLO un objeto JSON válido para mode='{mode}'.\n"
        "- No incluyas comentario, markdown ni texto fuera del JSON.\n"
        "- Si la información no está en el contexto, pon 'No disponible en el contexto' en el campo adecuado.\n"
    )

    # Si se especifican files, filtra/prioriza documentos que provienen de esos nombres
    if files_focus:
        focus_set = set(f.lower() for f in files_focus)
        prioritized = []
        others = []
        for d in context_docs:
            meta = d.metadata if hasattr(d, "metadata") else {}
            src = (meta.get("source") or "").lower()
            (prioritized if src in focus_set else others).append(d)
        context_docs = prioritized + others

    # Prepara el bloque de contexto ajustado al presupuesto de tokens
    base_suffix = (
        f"\n\nModo:\n{mode}\n\n"
        "Genera la salida JSON ahora:"
    )
    base_tokens = len(encoding.encode(SYSTEM_INSTRUCTIONS + formatting + base_suffix))
    allowed_tokens_for_context = max_model_tokens - reserved - base_tokens
    if allowed_tokens_for_context <= 0:
        allowed_tokens_for_context = max_model_tokens // 4

    used_tokens = 0
    context_texts = []

    for i, d in enumerate(context_docs):
        meta = d.metadata if hasattr(d, "metadata") else {}
        header = _format_chunk_header(meta, i)
        content = d.page_content or ""
        block = header + content
        tok_count = len(encoding.encode(block))

        if used_tokens + tok_count > allowed_tokens_for_context:
            remaining = allowed_tokens_for_context - used_tokens
            if remaining <= 0:
                break
            # truncado binario del contenido para encajar
            lo, hi = 0, len(content)
            best = 0
            while lo <= hi:
                mid = (lo + hi) // 2
                if len(encoding.encode(header + content[:mid])) <= remaining:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            if best > 0:
                truncated = content[:best]
                context_texts.append(f"{header}{truncated}")
                used_tokens += len(encoding.encode(header + truncated))
            break
        else:
            context_texts.append(block)
            used_tokens += tok_count

    context_block = "\n\n---\n\n".join(context_texts) if context_texts else ""

    files_line = f"Archivos a enfocar: {files_focus}\n" if files_focus else ""
    prompt = (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"{files_line}"
        f"Contexto recuperado (fragmentos con citas integradas):\n{context_block}\n\n"
        f"{formatting}\n"
        f"Pregunta del usuario:\n{question}"
        f"{base_suffix}"
    )
    return prompt

def answer_with_rag(
    question: str,
    k: Optional[int] = None,
    collection_name: Optional[str] = None,
    mode: str = "qa",
    files: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Ejecuta RAG con el modo deseado. El LLM debe devolver SIEMPRE JSON válido.
    """
    docs = get_relevant_docs(question, k=k, collection_name=collection_name)
    prompt = build_prompt(docs, question, mode=mode, files_focus=files)

    encoding = tiktoken.encoding_for_model(config.LLM_MODEL)
    tokens_used = len(encoding.encode(prompt))

    answer_json = llm.generate(prompt)  # Debe ser un string JSON válido
    return {
        "answer": answer_json,
        "source_documents": docs,
        "tokens_used": tokens_used,
        "mode": mode,
        "files": files or []
    }