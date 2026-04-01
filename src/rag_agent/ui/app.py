"""
app.py
======
Streamlit user interface for the Deep Learning RAG Interview Prep Agent.

Three-panel layout:
  - Left sidebar: Document ingestion and corpus browser
  - Centre: Document viewer
  - Right: Chat interface

API contract with the backend (agree this with Pipeline Engineer
before building anything):

  ingest(file_paths: list[Path]) -> IngestionResult
  list_documents() -> list[dict]
  get_document_chunks(source: str) -> list[DocumentChunk]
  chat(query: str, history: list[dict], filters: dict) -> AgentResponse

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import streamlit as st
from langchain_core.messages import HumanMessage

import rag_agent.agent.graph as agent_graph_module
import rag_agent.agent.nodes as agent_nodes_module
from rag_agent.agent.graph import get_compiled_graph
from rag_agent.config import get_settings
from rag_agent.corpus.chunker import DocumentChunker
from rag_agent.vectorstore.store import VectorStoreManager


# ---------------------------------------------------------------------------
# Cached Resources
# ---------------------------------------------------------------------------
# Use st.cache_resource for objects that should persist across reruns
# and be shared across all user sessions. This prevents re-initialising
# ChromaDB and reloading the embedding model on every button click.


@st.cache_resource
def get_vector_store() -> VectorStoreManager:
    """
    Return the singleton VectorStoreManager.

    Cached so ChromaDB connection is initialised once per application
    session, not on every Streamlit rerun.
    """
    return VectorStoreManager()


@st.cache_resource
def get_chunker() -> DocumentChunker:
    """Return the singleton DocumentChunker."""
    return DocumentChunker()


@st.cache_resource
def get_graph():
    """Return the compiled LangGraph agent."""
    return get_compiled_graph()


def reset_runtime_state(clear_chat_history: bool = False) -> None:
    """
    Clear cached backend resources and refresh session-level state.

    This gives the UI a safe recovery path after `.env` changes, collection
    switches, or stale cached resources in a long-running Streamlit session.
    """
    get_vector_store.clear()
    get_chunker.clear()
    get_graph.clear()
    get_settings.cache_clear()
    agent_graph_module.get_compiled_graph.cache_clear()
    agent_nodes_module._get_llm.cache_clear()
    agent_nodes_module._get_vector_store.cache_clear()

    st.session_state["ingested_documents"] = []
    st.session_state["last_ingestion_result"] = None
    st.session_state["selected_document"] = None
    st.session_state["topic_filter"] = None
    st.session_state["difficulty_filter"] = None
    st.session_state["thread_id"] = str(uuid4())

    if clear_chat_history:
        st.session_state["chat_history"] = []


# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------


def initialise_session_state() -> None:
    """
    Initialise all st.session_state keys on first run.

    Must be called at the top of main() before any UI is rendered.
    Without this, state keys referenced in callbacks will raise KeyError.

    Interview talking point: Streamlit reruns the entire script on every
    user interaction. session_state is the mechanism for persisting data
    (chat history, ingestion results) across reruns.
    """
    defaults = {
        "chat_history": [],           # list of {"role": "user"|"assistant", "content": str}
        "ingested_documents": [],     # list of dicts from list_documents()
        "selected_document": None,    # source filename currently in viewer
        "last_ingestion_result": None,
        "thread_id": str(uuid4()),    # LangGraph conversation thread
        "topic_filter": None,
        "difficulty_filter": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ---------------------------------------------------------------------------
# Ingestion Panel (Sidebar)
# ---------------------------------------------------------------------------


def render_ingestion_panel(
    store: VectorStoreManager,
    chunker: DocumentChunker,
) -> None:
    """
    Render the document ingestion panel in the sidebar.

    Allows multi-file upload of PDF and Markdown files. Displays
    ingestion results (chunks added, duplicates skipped, errors).
    Updates the ingested documents list after successful ingestion.

    Parameters
    ----------
    store : VectorStoreManager
    chunker : DocumentChunker
    """
    settings = get_settings()

    st.sidebar.header("📂 Corpus Ingestion")
    uploaded_files = st.sidebar.file_uploader(
        "Upload study materials",
        type=["pdf", "md"],
        accept_multiple_files=True,
    )

    if st.sidebar.button("Ingest Documents", disabled=not uploaded_files):
        corpus_dir = Path(settings.corpus_dir)
        corpus_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for uploaded_file in uploaded_files or []:
            target_path = corpus_dir / uploaded_file.name
            target_path.write_bytes(uploaded_file.getbuffer())
            saved_paths.append(target_path)

        with st.spinner("Chunking and ingesting documents..."):
            chunks = chunker.chunk_files(saved_paths)
            result = store.ingest(chunks)

        st.session_state["last_ingestion_result"] = result
        st.session_state["ingested_documents"] = store.list_documents()

    result = st.session_state.get("last_ingestion_result")
    if result is not None:
        if result.errors:
            st.sidebar.error(
                f"{result.ingested} chunks added, {result.skipped} skipped, "
                f"{len(result.errors)} errors"
            )
            for error in result.errors:
                st.sidebar.caption(error)
        elif result.ingested > 0 or result.skipped > 0:
            st.sidebar.success(
                f"{result.ingested} chunks added, {result.skipped} duplicates skipped"
            )

    if not st.session_state["ingested_documents"]:
        st.session_state["ingested_documents"] = store.list_documents()

    st.sidebar.info("Upload .pdf or .md files to populate the corpus.")
    st.sidebar.subheader("Ingested Documents")

    documents = st.session_state["ingested_documents"]
    if not documents:
        st.sidebar.caption("No documents ingested yet.")
        return

    for document in documents:
        st.sidebar.markdown(
            f"**{document['source']}**\n\n"
            f"Topic: `{document['topic']}`\n\n"
            f"Chunks: `{document['chunk_count']}`"
        )
        if st.sidebar.button(
            f"Remove {document['source']}",
            key=f"remove_{document['source']}",
        ):
            deleted_count = store.delete_document(document["source"])
            st.session_state["ingested_documents"] = store.list_documents()
            if st.session_state["selected_document"] == document["source"]:
                st.session_state["selected_document"] = None
            st.sidebar.warning(
                f"Removed {deleted_count} chunks from {document['source']}"
            )
            st.rerun()


def render_corpus_stats(store: VectorStoreManager) -> None:
    """
    Render a compact corpus health summary in the sidebar.

    Shows total chunks, topics covered, and whether bonus topics
    are present. Used during Hour 3 to demonstrate corpus completeness.

    Parameters
    ----------
    store : VectorStoreManager
    """
    stats = store.get_collection_stats()
    st.sidebar.subheader("Corpus Health")
    st.sidebar.metric("Total Chunks", stats["total_chunks"])
    st.sidebar.write(
        "Topics:",
        ", ".join(stats["topics"]) if stats["topics"] else "None yet",
    )
    if stats["bonus_topics_present"]:
        st.sidebar.success("Bonus topics present")
    else:
        st.sidebar.warning("Bonus topics not yet ingested")


def render_runtime_controls() -> None:
    """Render quick recovery actions for stale filters or cached backends."""
    st.sidebar.subheader("Runtime Controls")

    if st.sidebar.button("Reload Backend"):
        reset_runtime_state(clear_chat_history=False)
        st.rerun()

    if st.sidebar.button("Reset Chat And Filters"):
        st.session_state["chat_history"] = []
        st.session_state["topic_filter"] = None
        st.session_state["difficulty_filter"] = None
        st.session_state["thread_id"] = str(uuid4())
        st.rerun()


# ---------------------------------------------------------------------------
# Document Viewer Panel (Centre)
# ---------------------------------------------------------------------------


def render_document_viewer(store: VectorStoreManager) -> None:
    """
    Render the document viewer in the main centre column.

    Displays a selectable list of ingested documents. When a document
    is selected, renders its chunk content in a scrollable pane.

    Parameters
    ----------
    store : VectorStoreManager
    """
    st.subheader("📄 Document Viewer")

    documents = st.session_state["ingested_documents"] or store.list_documents()
    if not documents:
        st.info("Ingest documents using the sidebar to view content here.")
        return

    sources = [document["source"] for document in documents]
    if st.session_state["selected_document"] not in sources:
        st.session_state["selected_document"] = sources[0]

    selected_source = st.selectbox(
        "Select document",
        options=sources,
        index=sources.index(st.session_state["selected_document"]),
    )
    st.session_state["selected_document"] = selected_source

    chunks = store.get_document_chunks(selected_source)
    if not chunks:
        st.warning("No chunks found for the selected document.")
        return

    viewer_container = st.container(height=500)
    with viewer_container:
        for index, chunk in enumerate(chunks, start=1):
            st.caption(
                f"Chunk {index} | {chunk.metadata.topic} | "
                f"{chunk.metadata.difficulty} | {chunk.metadata.type}"
            )
            st.write(chunk.chunk_text)
            st.divider()

    st.caption(f"{len(chunks)} chunks in {selected_source}")


# ---------------------------------------------------------------------------
# Chat Interface Panel (Right)
# ---------------------------------------------------------------------------


def render_chat_interface(graph) -> None:
    """
    Render the chat interface in the right column.

    Supports multi-turn conversation with the LangGraph agent.
    Displays source citations with every response.
    Shows a clear "no relevant context" indicator when the
    hallucination guard fires.

    Parameters
    ----------
    graph : CompiledStateGraph
        The compiled LangGraph agent from get_compiled_graph().
    """
    st.subheader("💬 Interview Prep Chat")

    # Filters
    documents = st.session_state["ingested_documents"] or []
    available_topics = sorted({document["topic"] for document in documents})
    topic_options = ["All"] + available_topics
    difficulty_options = ["All", "beginner", "intermediate", "advanced"]

    col_topic, col_diff = st.columns(2)
    with col_topic:
        current_topic = st.session_state["topic_filter"] or "All"
        selected_topic = st.selectbox(
            "Topic",
            options=topic_options,
            index=topic_options.index(current_topic)
            if current_topic in topic_options
            else 0,
        )
        st.session_state["topic_filter"] = None if selected_topic == "All" else selected_topic
    with col_diff:
        current_difficulty = st.session_state["difficulty_filter"] or "All"
        selected_difficulty = st.selectbox(
            "Difficulty",
            options=difficulty_options,
            index=difficulty_options.index(current_difficulty)
            if current_difficulty in difficulty_options
            else 0,
        )
        st.session_state["difficulty_filter"] = (
            None if selected_difficulty == "All" else selected_difficulty
        )

    active_topic = st.session_state["topic_filter"] or "All"
    active_difficulty = st.session_state["difficulty_filter"] or "All"
    st.caption(
        f"Active filters: topic = {active_topic}, difficulty = {active_difficulty}"
    )

    # Chat history display
    chat_container = st.container(height=400)
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("rewritten_query"):
                    with st.expander("Retrieval Debug"):
                        st.caption(f"Rewritten query: {message['rewritten_query']}")
                        st.caption(
                            "Filters used: "
                            f"topic = {message.get('topic_filter') or 'All'}, "
                            f"difficulty = {message.get('difficulty_filter') or 'All'}"
                        )
                if message.get("sources"):
                    with st.expander("📎 Sources"):
                        for source in message["sources"]:
                            st.caption(source)
                if message.get("no_context_found"):
                    st.warning("⚠️ No relevant content found in corpus.")
                    if message.get("topic_filter") or message.get("difficulty_filter"):
                        st.info(
                            "The active filters may have narrowed retrieval too much. "
                            "Try setting both filters to All or use Reset Chat And Filters."
                        )

    query = st.chat_input("Ask about a deep learning topic or request an interview question...")
    if not query:
        return

    if not query.strip():
        st.warning("Please enter a non-empty question.")
        return

    st.session_state.chat_history.append({"role": "user", "content": query})

    graph_input = {
        "messages": [HumanMessage(content=query)],
        "topic_filter": st.session_state["topic_filter"],
        "difficulty_filter": st.session_state["difficulty_filter"],
    }
    config = {"configurable": {"thread_id": st.session_state["thread_id"]}}

    with st.spinner("Thinking..."):
        result = graph.invoke(graph_input, config=config)

    response = result.get("final_response")
    if response is None:
        fallback_answer = (
            "The agent did not return a structured response. Please try the query again."
        )
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": fallback_answer,
                "sources": [],
                "no_context_found": True,
                "rewritten_query": "",
                "topic_filter": st.session_state["topic_filter"],
                "difficulty_filter": st.session_state["difficulty_filter"],
            }
        )
    else:
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": response.answer,
                "sources": response.sources,
                "no_context_found": response.no_context_found,
                "rewritten_query": response.rewritten_query,
                "topic_filter": st.session_state["topic_filter"],
                "difficulty_filter": st.session_state["difficulty_filter"],
            }
        )

    st.rerun()


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Application entry point.

    Sets page config, initialises session state, instantiates shared
    resources, and renders all UI panels.

    Run with: uv run streamlit run src/rag_agent/ui/app.py
    """
    settings = get_settings()

    st.set_page_config(
        page_title=settings.app_title,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(f"🧠 {settings.app_title}")
    st.caption(
        "RAG-powered interview preparation — built with LangChain, LangGraph, and ChromaDB"
    )

    initialise_session_state()

    # Instantiate shared backend resources
    store = get_vector_store()
    chunker = get_chunker()
    graph = get_graph()

    # Sidebar
    render_ingestion_panel(store, chunker)
    render_corpus_stats(store)
    render_runtime_controls()

    # Main content area — two columns
    viewer_col, chat_col = st.columns([1, 1], gap="large")

    with viewer_col:
        render_document_viewer(store)

    with chat_col:
        render_chat_interface(graph)


if __name__ == "__main__":
    main()
