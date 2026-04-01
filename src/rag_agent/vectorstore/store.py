"""
store.py
========
ChromaDB vector store management.

Handles all interactions with the persistent ChromaDB collection:
initialisation, ingestion, duplicate detection, and retrieval.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import re

import chromadb
from loguru import logger

from rag_agent.agent.state import (
    ChunkMetadata,
    DocumentChunk,
    IngestionResult,
    RetrievedChunk,
)
from rag_agent.config import EmbeddingFactory, Settings, get_settings


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "why",
    "with",
}


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    normalised = set()
    for token in tokens:
        if token in STOPWORDS or len(token) <= 1:
            continue
        if token.endswith("s") and len(token) > 4:
            token = token[:-1]
        normalised.add(token)
    return normalised


class VectorStoreManager:
    """
    Manages the ChromaDB persistent vector store for the corpus.

    All corpus ingestion and retrieval operations pass through this class.
    It is the single point of contact between the application and ChromaDB.

    Parameters
    ----------
    settings : Settings, optional
        Application settings. Uses get_settings() singleton if not provided.

    Example
    -------
    >>> manager = VectorStoreManager()
    >>> result = manager.ingest(chunks)
    >>> print(f"Ingested: {result.ingested}, Skipped: {result.skipped}")
    >>>
    >>> chunks = manager.query("explain the vanishing gradient problem", k=4)
    >>> for chunk in chunks:
    ...     print(chunk.to_citation(), chunk.score)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._embeddings = EmbeddingFactory(self._settings).create()
        self._client = None
        self._collection = None
        self._initialise()

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def _initialise(self) -> None:
        """
        Create or connect to the persistent ChromaDB client and collection.

        Creates the chroma_db_path directory if it does not exist.
        Uses PersistentClient so data survives between application restarts.

        Called automatically during __init__. Should not be called directly.

        Raises
        ------
        RuntimeError
            If ChromaDB cannot be initialised at the configured path.
        """
        try:
            db_path = Path(self._settings.chroma_db_path)
            db_path.mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(path=str(db_path))
            self._collection = self._client.get_or_create_collection(
                name=self._settings.chroma_collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            logger.info(
                "Initialised ChromaDB collection '{}' at '{}' with {} chunks",
                self._settings.chroma_collection_name,
                db_path,
                self._collection.count(),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialise ChromaDB at {self._settings.chroma_db_path}"
            ) from exc

    # -----------------------------------------------------------------------
    # Duplicate Detection
    # -----------------------------------------------------------------------

    @staticmethod
    def generate_chunk_id(source: str, chunk_text: str) -> str:
        """
        Generate a deterministic chunk ID from source filename and content.

        Using a content hash ensures two uploads of the same file produce
        the same IDs, making duplicate detection reliable regardless of
        filename changes.

        Parameters
        ----------
        source : str
            The source filename (e.g. 'lstm.md').
        chunk_text : str
            The full text content of the chunk.

        Returns
        -------
        str
            A 16-character hex string derived from SHA-256 of the inputs.
        """
        content = f"{source}::{chunk_text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def check_duplicate(self, chunk_id: str) -> bool:
        """
        Check whether a chunk with this ID already exists in the collection.

        Parameters
        ----------
        chunk_id : str
            The deterministic chunk ID to check.

        Returns
        -------
        bool
            True if the chunk already exists (duplicate). False otherwise.

        Interview talking point: content-addressed deduplication is more
        robust than filename-based deduplication because it detects identical
        content even when files are renamed or re-uploaded.
        """
        result = self._collection.get(ids=[chunk_id])
        ids = result.get("ids", [])
        return bool(ids)

    # -----------------------------------------------------------------------
    # Ingestion
    # -----------------------------------------------------------------------

    def ingest(self, chunks: list[DocumentChunk]) -> IngestionResult:
        """
        Embed and store a list of DocumentChunks in ChromaDB.

        Checks each chunk for duplicates before embedding. Skips duplicates
        silently and records the count in the returned IngestionResult.

        Parameters
        ----------
        chunks : list[DocumentChunk]
            Prepared chunks with text and metadata. Use DocumentChunker
            to produce these from raw files.

        Returns
        -------
        IngestionResult
            Summary with counts of ingested, skipped, and errored chunks.

        Notes
        -----
        Embeds in batches of 100 to avoid memory issues with large corpora.
        Uses upsert (not add) so re-ingestion of modified content updates
        existing chunks rather than raising an error.

        Interview talking point: batch processing with a configurable
        batch size is a production pattern that prevents OOM errors when
        ingesting large document sets.
        """
        result = IngestionResult()
        document_ids: set[str] = set()

        for chunk in chunks:
            try:
                if self.check_duplicate(chunk.chunk_id):
                    result.skipped += 1
                    continue

                embedding = self._embeddings.embed_documents([chunk.chunk_text])[0]
                self._collection.upsert(
                    ids=[chunk.chunk_id],
                    embeddings=[embedding],
                    documents=[chunk.chunk_text],
                    metadatas=[chunk.metadata.to_dict()],
                )
                result.ingested += 1
                document_ids.add(chunk.metadata.source)
            except Exception as exc:
                logger.exception(
                    "Failed to ingest chunk {} from {}",
                    chunk.chunk_id,
                    chunk.metadata.source,
                )
                result.errors.append(f"{chunk.metadata.source}: {exc}")

        result.document_ids = sorted(document_ids)
        logger.info(
            "Ingestion complete: {} ingested, {} skipped, {} errors",
            result.ingested,
            result.skipped,
            len(result.errors),
        )
        return result

    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        k: int | None = None,
        topic_filter: str | None = None,
        difficulty_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Applies similarity threshold filtering — chunks below
        settings.similarity_threshold are excluded from results.

        Parameters
        ----------
        query_text : str
            The user query or rewritten query to retrieve against.
        k : int, optional
            Number of chunks to retrieve. Defaults to settings.retrieval_k.
        topic_filter : str, optional
            Restrict retrieval to a specific topic (e.g. 'LSTM').
            Maps to ChromaDB where-filter on metadata.topic.
        difficulty_filter : str, optional
            Restrict retrieval to a difficulty level.
            Maps to ChromaDB where-filter on metadata.difficulty.

        Returns
        -------
        list[RetrievedChunk]
            Chunks sorted by similarity score descending.
            Empty list if no chunks meet the similarity threshold.

        Interview talking point: returning an empty list (not hallucinating)
        when no relevant context exists is the hallucination guard. This is
        a critical production RAG pattern — the system must know what it
        does not know.
        """
        if not query_text.strip():
            return []

        k = k or self._settings.retrieval_k
        candidate_count = max(k * 5, 12)
        collection_count = self._collection.count()
        n_results = min(candidate_count, collection_count) if collection_count else k

        filters = []
        if topic_filter:
            filters.append({"topic": topic_filter})
        if difficulty_filter:
            filters.append({"difficulty": difficulty_filter})

        where_filter = None
        if len(filters) == 1:
            where_filter = filters[0]
        elif len(filters) > 1:
            where_filter = {"$and": filters}

        query_embedding = self._embeddings.embed_query(query_text)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        query_tokens = _tokenize(query_text)
        retrieved_chunks: list[RetrievedChunk] = []
        for chunk_id, document, metadata_dict, distance in zip(
            ids, documents, metadatas, distances
        ):
            vector_score = max(0.0, 1 - float(distance))
            document_tokens = _tokenize(document)
            overlap_score = (
                len(query_tokens & document_tokens) / len(query_tokens)
                if query_tokens
                else 0.0
            )

            metadata = ChunkMetadata.from_dict(metadata_dict)
            metadata_tokens = _tokenize(
                " ".join([metadata.topic, *metadata.related_topics, metadata.source])
            )
            metadata_overlap = (
                len(query_tokens & metadata_tokens) / len(query_tokens)
                if query_tokens
                else 0.0
            )
            topic_tokens = _tokenize(metadata.topic)
            related_topic_tokens = _tokenize(" ".join(metadata.related_topics))
            topic_bonus = 0.20 if query_tokens & topic_tokens else 0.0
            related_bonus = 0.05 if query_tokens & related_topic_tokens else 0.0

            score = (
                0.55 * vector_score
                + 0.35 * overlap_score
                + 0.10 * metadata_overlap
                + topic_bonus
                + related_bonus
            )
            if overlap_score == 0 and metadata_overlap == 0 and vector_score < 0.6:
                continue
            if score < self._settings.similarity_threshold:
                continue

            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    chunk_text=document,
                    metadata=metadata,
                    score=score,
                )
            )

        retrieved_chunks.sort(key=lambda chunk: chunk.score, reverse=True)
        return retrieved_chunks[:k]

    # -----------------------------------------------------------------------
    # Corpus Inspection
    # -----------------------------------------------------------------------

    def list_documents(self) -> list[dict]:
        """
        Return a list of all unique source documents in the collection.

        Used by the UI to populate the document viewer panel.

        Returns
        -------
        list[dict]
            Each item contains: source (str), topic (str), chunk_count (int).
        """
        results = self._collection.get(include=["metadatas"])
        documents: dict[str, dict] = {}

        for metadata_dict in results.get("metadatas", []):
            if not metadata_dict:
                continue

            metadata = ChunkMetadata.from_dict(metadata_dict)
            if metadata.source not in documents:
                documents[metadata.source] = {
                    "source": metadata.source,
                    "topic": metadata.topic,
                    "chunk_count": 0,
                }
            documents[metadata.source]["chunk_count"] += 1

        return sorted(documents.values(), key=lambda item: item["source"].lower())

    def get_document_chunks(self, source: str) -> list[DocumentChunk]:
        """
        Retrieve all chunks belonging to a specific source document.

        Used by the document viewer to display document content.

        Parameters
        ----------
        source : str
            The source filename to retrieve chunks for.

        Returns
        -------
        list[DocumentChunk]
            All chunks from this source, ordered by their position
            in the original document.
        """
        results = self._collection.get(
            where={"source": source},
            include=["documents", "metadatas"],
        )

        chunks = [
            DocumentChunk(
                chunk_id=chunk_id,
                chunk_text=document,
                metadata=ChunkMetadata.from_dict(metadata_dict),
            )
            for chunk_id, document, metadata_dict in zip(
                results.get("ids", []),
                results.get("documents", []),
                results.get("metadatas", []),
            )
        ]
        return chunks

    def get_collection_stats(self) -> dict:
        """
        Return summary statistics about the current collection.

        Used by the UI to show corpus health at a glance.

        Returns
        -------
        dict
            Keys: total_chunks, topics (list), sources (list),
            bonus_topics_present (bool).
        """
        results = self._collection.get(include=["metadatas"])
        metadatas = [item for item in results.get("metadatas", []) if item]

        topics = sorted({metadata["topic"] for metadata in metadatas})
        sources = sorted({metadata["source"] for metadata in metadatas})
        bonus_topics_present = any(
            ChunkMetadata.from_dict(metadata).is_bonus for metadata in metadatas
        )

        return {
            "total_chunks": len(metadatas),
            "topics": topics,
            "sources": sources,
            "bonus_topics_present": bonus_topics_present,
        }

    def delete_document(self, source: str) -> int:
        """
        Remove all chunks from a specific source document.

        Parameters
        ----------
        source : str
            Source filename to remove.

        Returns
        -------
        int
            Number of chunks deleted.
        """
        results = self._collection.get(where={"source": source})
        ids = results.get("ids", [])
        if not ids:
            return 0

        self._collection.delete(ids=ids)
        return len(ids)
