"""
nodes.py
========
LangGraph node functions for the RAG interview preparation agent.

Each function in this module is a node in the agent state graph.
Nodes receive the current AgentState, perform their operation,
and return a dict of state fields to update.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

import json
from functools import lru_cache

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from rag_agent.agent.prompts import (
    NO_CONTEXT_RESPONSE,
    QUESTION_GENERATION_PROMPT,
    QUERY_REWRITE_PROMPT,
    SYSTEM_PROMPT,
)
from rag_agent.agent.state import AgentResponse, AgentState
from rag_agent.config import LLMFactory, get_settings
from rag_agent.vectorstore.store import VectorStoreManager


@lru_cache(maxsize=1)
def _get_llm():
    return LLMFactory(get_settings()).create()


@lru_cache(maxsize=1)
def _get_vector_store() -> VectorStoreManager:
    return VectorStoreManager(get_settings())


def _trim_history(messages: list, max_context_tokens: int) -> list:
    """Approximate token trimming using word count for predictable local behavior."""
    trimmed_messages = []
    token_count = 0

    for message in reversed(messages):
        content = getattr(message, "content", "")
        message_tokens = len(str(content).split())
        if token_count + message_tokens > max_context_tokens:
            break
        trimmed_messages.append(message)
        token_count += message_tokens

    return list(reversed(trimmed_messages))


def _state_value(state, key: str, default=None):
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def _is_question_generation_request(query: str) -> bool:
    lowered = query.lower()
    return "question" in lowered and (
        "generate" in lowered or "interview" in lowered or "quiz" in lowered
    )


def _format_generated_question(payload: dict) -> str:
    sections = []
    question = payload.get("question")
    if question:
        sections.append(f"Interview Question:\n{question}")

    model_answer = payload.get("model_answer")
    if model_answer:
        sections.append(f"Model Answer:\n{model_answer}")

    follow_up = payload.get("follow_up")
    if follow_up:
        sections.append(f"Follow-up:\n{follow_up}")

    return "\n\n".join(sections) if sections else json.dumps(payload, indent=2)


def _fallback_answer(state: AgentState, citations: list[str]) -> str:
    top_chunks = _state_value(state, "retrieved_chunks", [])[:2]
    summary_parts = [chunk.chunk_text for chunk in top_chunks]
    summary = "\n\n".join(summary_parts)
    return (
        "LLM provider unavailable, so this response was assembled directly from the "
        "retrieved study material.\n\n"
        f"Relevant context for '{_state_value(state, 'original_query', '')}':\n\n"
        f"{summary}\n\n"
        f"Sources: {', '.join(citations)}"
    )


def _fallback_generated_question(state: AgentState, citations: list[str]) -> str:
    top_chunk = _state_value(state, "retrieved_chunks", [])[0]
    follow_up_target = (
        top_chunk.metadata.related_topics[0]
        if top_chunk.metadata.related_topics
        else top_chunk.metadata.topic
    )
    return (
        f"Interview Question:\nExplain {top_chunk.metadata.topic} in the context of "
        "deep learning, and describe why it matters in model behavior.\n\n"
        f"Model Answer:\n{top_chunk.chunk_text}\n\n"
        f"Follow-up:\nHow does {top_chunk.metadata.topic} relate to {follow_up_target}?\n\n"
        f"Sources: {', '.join(citations)}"
    )


# ---------------------------------------------------------------------------
# Node: Query Rewriter
# ---------------------------------------------------------------------------


def query_rewrite_node(state: AgentState) -> dict:
    """
    Rewrite the user's query to maximise retrieval effectiveness.

    Natural language questions are often poorly suited for vector
    similarity search. This node rephrases the query into a form
    that produces better embedding matches against the corpus.

    Example
    -------
    Input:  "I'm confused about how LSTMs remember things long-term"
    Output: "LSTM long-term memory cell state forget gate mechanism"

    Interview talking point: query rewriting is a production RAG pattern
    that significantly improves retrieval recall. It acknowledges that
    users do not phrase queries the way documents are written.

    Parameters
    ----------
    state : AgentState
        Current graph state. Reads: messages (for context).

    Returns
    -------
    dict
        Updates: original_query, rewritten_query.
    """
    messages = _state_value(state, "messages", [])
    latest_human_message = next(
        (
            message
            for message in reversed(messages)
            if isinstance(message, HumanMessage)
        ),
        None,
    )
    original_query = latest_human_message.content.strip() if latest_human_message else ""

    if not original_query:
        return {"original_query": "", "rewritten_query": ""}

    try:
        llm = _get_llm()
        rewrite_prompt = QUERY_REWRITE_PROMPT.format(original_query=original_query)
        rewritten = llm.invoke(
            [
                SystemMessage(
                    content="Rewrite user questions for retrieval only. Return only the rewritten query."
                ),
                HumanMessage(content=rewrite_prompt),
            ]
        )
        rewritten_query = getattr(rewritten, "content", original_query).strip()
        return {
            "original_query": original_query,
            "rewritten_query": rewritten_query or original_query,
        }
    except Exception:
        return {
            "original_query": original_query,
            "rewritten_query": original_query,
        }


# ---------------------------------------------------------------------------
# Node: Retriever
# ---------------------------------------------------------------------------


def retrieval_node(state: AgentState) -> dict:
    """
    Retrieve relevant chunks from ChromaDB based on the rewritten query.

    Sets the no_context_found flag if no chunks meet the similarity
    threshold. This flag is checked by generation_node to trigger
    the hallucination guard.

    Interview talking point: separating retrieval into its own node
    makes it independently testable and replaceable — you could swap
    ChromaDB for Pinecone or Weaviate by changing only this node.

    Parameters
    ----------
    state : AgentState
        Current graph state.
        Reads: rewritten_query, topic_filter, difficulty_filter.

    Returns
    -------
    dict
        Updates: retrieved_chunks, no_context_found.
    """
    manager = _get_vector_store()
    query_text = _state_value(state, "rewritten_query", "") or _state_value(
        state, "original_query", ""
    )
    chunks = manager.query(
        query_text=query_text,
        topic_filter=_state_value(state, "topic_filter"),
        difficulty_filter=_state_value(state, "difficulty_filter"),
    )

    if not chunks:
        return {
            "retrieved_chunks": [],
            "no_context_found": True,
        }

    return {
        "retrieved_chunks": chunks,
        "no_context_found": False,
    }


# ---------------------------------------------------------------------------
# Node: Generator
# ---------------------------------------------------------------------------


def generation_node(state: AgentState) -> dict:
    """
    Generate the final response using retrieved chunks as context.

    Implements the hallucination guard: if no_context_found is True,
    returns a clear "no relevant context" message rather than allowing
    the LLM to answer from parametric memory.

    Implements token-aware conversation memory trimming: when the
    message history approaches max_context_tokens, the oldest
    non-system messages are removed.

    Interview talking point: the hallucination guard is the most
    commonly asked about production RAG pattern. Interviewers want
    to know how you prevent the model from confidently making up
    information when the retrieval step finds nothing relevant.

    Parameters
    ----------
    state : AgentState
        Current graph state.
        Reads: retrieved_chunks, no_context_found, messages,
               original_query, topic_filter.

    Returns
    -------
    dict
        Updates: final_response, messages (with new AIMessage appended).
    """
    settings = get_settings()

    # ---- Hallucination Guard -----------------------------------------------
    if _state_value(state, "no_context_found", False):
        no_context_message = NO_CONTEXT_RESPONSE
        response = AgentResponse(
            answer=no_context_message,
            sources=[],
            confidence=0.0,
            no_context_found=True,
            rewritten_query=_state_value(state, "rewritten_query", ""),
        )
        return {
            "final_response": response,
            "messages": [AIMessage(content=no_context_message)],
        }

    context_blocks = []
    citations = []
    retrieved_chunks = _state_value(state, "retrieved_chunks", [])
    for chunk in retrieved_chunks:
        citation = f"[SOURCE: {chunk.metadata.topic} | {chunk.metadata.source}]"
        citations.append(citation)
        context_blocks.append(f"{citation}\n{chunk.chunk_text}")

    context_string = "\n\n".join(context_blocks)
    unique_citations = list(dict.fromkeys(citations))
    confidence = sum(chunk.score for chunk in retrieved_chunks) / len(
        retrieved_chunks
    )
    messages = _state_value(state, "messages", [])
    trimmed_history = _trim_history(messages, settings.max_context_tokens)
    original_query = _state_value(state, "original_query", "") or (
        messages[-1].content if messages else ""
    )

    if _is_question_generation_request(original_query):
        difficulty = (
            _state_value(state, "difficulty_filter")
            or retrieved_chunks[0].metadata.difficulty
            or "intermediate"
        )
        try:
            llm = _get_llm()
            prompt = QUESTION_GENERATION_PROMPT.format(
                context=context_string,
                difficulty=difficulty,
            )
            llm_response = llm.invoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ]
            )
            answer_text = getattr(llm_response, "content", str(llm_response))

            try:
                payload = json.loads(answer_text)
                answer_text = _format_generated_question(payload)
                generated_sources = payload.get("source_citations")
                if isinstance(generated_sources, list) and generated_sources:
                    unique_citations = generated_sources
            except Exception:
                pass
        except Exception:
            answer_text = _fallback_generated_question(state, unique_citations)
    else:
        try:
            llm = _get_llm()
            llm_response = llm.invoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    SystemMessage(content=f"Retrieved context:\n\n{context_string}"),
                    *trimmed_history,
                ]
            )
            answer_text = getattr(llm_response, "content", str(llm_response))
        except Exception:
            answer_text = _fallback_answer(state, unique_citations)

    response = AgentResponse(
        answer=answer_text,
        sources=unique_citations,
        confidence=confidence,
        no_context_found=False,
        rewritten_query=_state_value(state, "rewritten_query", ""),
    )
    new_ai_message = AIMessage(content=answer_text)

    return {
        "final_response": response,
        "messages": [new_ai_message],
    }


# ---------------------------------------------------------------------------
# Routing Function
# ---------------------------------------------------------------------------


def should_retry_retrieval(state: AgentState) -> str:
    """
    Conditional edge function: decide whether to retry retrieval or generate.

    Called by the graph after retrieval_node. If no context was found,
    the graph routes back to query_rewrite_node for one retry with a
    broader query before triggering the hallucination guard.

    Interview talking point: conditional edges in LangGraph enable
    agentic behaviour — the graph makes decisions about its own
    execution path rather than following a fixed sequence.

    Parameters
    ----------
    state : AgentState
        Current graph state. Reads: no_context_found, retrieved_chunks.

    Returns
    -------
    str
        "generate" — proceed to generation_node.
        "end"      — skip generation, return no_context response directly.

    Notes
    -----
    Retry logic should be limited to one attempt to prevent infinite loops.
    Track retry count in AgentState if implementing retry behaviour.
    """
    return "generate"
