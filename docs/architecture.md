# System Architecture
## Team: Rahul Pogula + Asha Sri Sayana
## Date: March 24, 2026
## Members and Roles:
- Corpus Architect + UX Lead: Asha Sri Sayana
- Pipeline Engineer + QA Lead: Rahul Pogula
- Prompt Engineer: Shared by both team members

---

## Architecture Diagram

```text
                    INGESTION FLOW

  .md / .pdf files uploaded in Streamlit
                  |
                  v
        DocumentChunker loads file
                  |
                  v
     Markdown/PDF chunking + metadata inference
                  |
                  v
  chunk_id = SHA256(source + chunk_text)[:16]
                  |
                  v
      VectorStoreManager.check_duplicate()
          |                         |
          | duplicate              | new chunk
          v                         v
       skip chunk          embed with all-MiniLM-L6-v2
                                      |
                                      v
                        ChromaDB PersistentClient collection
                               (cosine similarity)


                     QUERY / CHAT FLOW

      User question in Streamlit chat + optional filters
                          |
                          v
              LangGraph thread (thread_id)
                          |
                          v
                 query_rewrite_node
                          |
                          v
                   retrieval_node
                          |
          +---------------+----------------+
          |                                |
          | chunks above threshold         | no chunks above threshold
          v                                v
    generation_node                 hallucination guard response
          |
          v
  cited answer + confidence + sources
          |
          v
       Streamlit chat UI


               MEMORY / STATE ACROSS TURNS

  LangGraph MemorySaver keeps per-thread state
  Streamlit session_state keeps UI state:
  chat_history, selected_document, filters, ingestion results, thread_id
```

The diagram shows:
- [x] How a corpus file becomes a chunk
- [x] How a chunk becomes an embedding
- [x] How duplicate detection fires
- [x] How a user query flows through LangGraph to a response
- [x] Where the hallucination guard sits in the graph
- [x] How conversation memory is maintained across turns

---

## Component Descriptions

### Corpus Layer

- **Source files location:** `data/corpus/`
- **File formats used:** Markdown (`.md`) is currently ingested and validated in the working system. PDF ingestion support is implemented in the pipeline and reserved for landmark paper expansion.

- **Landmark papers identified for ingestion:**
  - Rumelhart, Hinton & Williams (1986) for backpropagation / ANN
  - LeCun et al. (1998) for CNN / LeNet
  - Elman (1990) for RNN
  - Hochreiter & Schmidhuber (1997) for LSTM
  - Sutskever, Vinyals & Le (2014) for Seq2Seq
  - Hinton & Salakhutdinov (2006) for Autoencoder

- **Chunking strategy:**
  Header-aware chunking for Markdown and recursive text splitting for PDFs.
  Default chunk target is 512 characters with 50 characters of overlap.
  This keeps chunks small enough for precise retrieval while preserving enough
  context for one interview-style explanation per chunk. Markdown headers are
  used as natural semantic boundaries before any fallback character splitting.

- **Metadata schema:**
  Every chunk carries the same metadata so retrieval, filtering, citation, and
  demo explanations stay consistent.

  | Field | Type | Purpose |
  |---|---|---|
  | topic | string | Primary deep learning topic used for filtering and citation |
  | difficulty | string | Controls question difficulty and lets the UI filter results |
  | type | string | Labels the chunk as concept, architecture, comparison, training process, or math foundation |
  | source | string | Original filename used for traceability and source citation |
  | related_topics | list | Helps explain conceptual links and supports richer retrieval reasoning |
  | is_bonus | bool | Distinguishes optional bonus topics such as GAN, SOM, and Boltzmann Machines |

- **Duplicate detection approach:**
  Chunk IDs are generated from a deterministic SHA-256 hash of `source + chunk_text`,
  truncated to 16 hex characters. A content hash is more reliable than a filename
  because the same file can be renamed and re-uploaded while still representing
  identical content. If the content is unchanged, the ID stays unchanged and the
  duplicate guard fires correctly.

- **Target corpus coverage for submission:**
  - [x] ANN
  - [x] CNN
  - [x] RNN
  - [x] LSTM
  - [x] Seq2Seq
  - [x] Autoencoder
  - [ ] SOM *(bonus)*
  - [ ] Boltzmann Machine *(bonus)*
  - [ ] GAN *(bonus)*

---

### Vector Store Layer

- **Database:** ChromaDB using `PersistentClient`
- **Local persistence path:** `./data/chroma_db`

- **Embedding model:**
  `all-MiniLM-L6-v2` via `sentence-transformers` and LangChain community embeddings

- **Why this embedding model:**
  It is fast, local, free, and lightweight enough to run on a student laptop
  without a paid API dependency. It offers a strong speed-to-quality tradeoff
  for class-scale corpora and avoids sending the study material to a third-party
  embedding API.

- **Similarity metric:**
  Cosine similarity. ChromaDB is initialised with `{"hnsw:space": "cosine"}` so
  score comparisons are intuitive and easy to explain in the demo.

- **Collection name:**
  `deep_learning_corpus_groq_live`

- **Retrieval k:**
  `4` chunks per query. The implementation retrieves a larger candidate set
  internally and then reranks down to the top 4.

- **Similarity threshold:**
  `0.15` in the current validated build. A lower threshold gave better separation
  between relevant and irrelevant queries while still allowing the hallucination
  guard to fire on clearly off-topic prompts.

- **Metadata filtering:**
  Users can filter by `topic` and `difficulty` from the UI. These become ChromaDB
  `where` filters on metadata before retrieval results are returned to the graph.

---

### Agent Layer

- **Framework:** LangGraph

- **Graph nodes:**

  | Node | Responsibility |
  |---|---|
  | query_rewrite_node | Rewrites the latest user question into a short keyword-dense retrieval query |
  | retrieval_node | Queries ChromaDB using the rewritten query plus optional topic and difficulty filters |
  | generation_node | Produces the final cited answer or returns the no-context response when retrieval fails |

- **Conditional edges:**
  The retrieval step determines whether relevant context exists. If relevant
  chunks are found, the graph proceeds to answer generation. If no chunks pass
  the similarity threshold, the system returns the hallucination-guard path
  instead of allowing the model to improvise from background knowledge.

- **Hallucination guard:**
  Exact system response:

  `I was unable to find relevant information in the study corpus for your query. This may mean the topic is not yet covered in the corpus, your query needs to be more specific, or the corpus needs more content on this area.`

- **Query rewriting example:**
  - Raw query: `I'm confused about how LSTMs remember things`
  - Rewritten query: `LSTM long-term memory cell state forget gate mechanism`

- **Conversation memory:**
  Multi-turn memory is maintained in LangGraph through `MessagesState` and
  `MemorySaver`, keyed by `thread_id`. The Streamlit layer also stores chat
  history in `st.session_state`. When the message window grows too large, older
  conversation history is trimmed using `trim_messages(..., strategy="last")`
  so recent context is preserved without exceeding the configured context budget.

- **LLM provider:**
  Groq using `llama-3.1-8b-instant`. Validated end-to-end with a live Groq API
  key. The LLM factory also supports Ollama and LM Studio as drop-in alternatives
  without changing the graph design.

- **Why this provider:**
  Groq is the fastest path to a working hosted model for a class project because
  it avoids local GPU setup, has low latency, and matches the repo's recommended
  path. If Groq access is unavailable, Ollama or LM Studio can be swapped in
  without changing the graph design because the LLM factory abstracts provider
  selection behind the same interface.

---

### Prompt Layer

- **System prompt summary:**
  The agent behaves like a senior machine learning engineer running an interview
  prep session. It must answer only from retrieved context, cite sources on every
  factual claim, refuse unsupported answers, and stay technically rigorous while
  still being encouraging.

- **Question generation prompt:**
  Inputs: retrieved context plus target difficulty.
  Returns: a JSON object containing one interview question, topic, difficulty,
  model answer, follow-up question, and source citations.

- **Answer evaluation prompt:**
  Inputs: question, candidate answer, and source context.
  Returns: a JSON object containing a `0-10` score, what was correct, what was
  missing, an ideal answer, an interview verdict, and a coaching tip.

- **JSON reliability:**
  Every structured prompt ends with a strict instruction equivalent to:
  `Respond with the JSON object only. No preamble, explanation, or markdown code fences.`
  This reduces parsing failures and makes downstream automation safer.

- **Failure modes identified:**
  - System prompt: the model may answer from general knowledge instead of retrieved context. Mitigation: explicit grounding rules plus a no-context fallback path.
  - Question generation: the model may output trivial recall questions or malformed JSON. Mitigation: force open-ended questions and JSON-only output.
  - Answer evaluation: the model may score too generously. Mitigation: include an explicit rubric and require explanation of missing concepts.
  - Query rewrite: the rewritten query may become too narrow or drop important terms. Mitigation: keep the rewrite short, technical, and fall back to the original query if rewriting fails.

---

### Interface Layer

- **Framework:** Streamlit
- **Deployment platform:** Streamlit Community Cloud
- **Public URL:** Pending deployment after final integration

- **Ingestion panel features:**
  Multi-file upload for `.md` and `.pdf` files, an ingest button, status messages
  for chunks added and duplicates skipped, a list of ingested source documents,
  and a remove-document action for cleanup during demos and testing.

- **Document viewer features:**
  A document selector shows ingested sources. Selecting a source displays its
  chunks in order with metadata labels such as topic, difficulty, and chunk type.
  This makes corpus quality visible during the demo rather than hiding retrieval
  behind the chat interface.

- **Chat panel features:**
  The chat panel supports multi-turn interaction, optional topic and difficulty
  filters, visible source citations for every response, rewritten-query debugging
  if needed, and a clear no-context indicator when the hallucination guard fires.

- **Session state keys:**

  | Key | Stores |
  |---|---|
  | chat_history | User and assistant messages shown in the Streamlit chat panel |
  | ingested_documents | Document summaries returned by `list_documents()` |
  | selected_document | The source currently displayed in the document viewer |
  | last_ingestion_result | Most recent ingestion status for UI feedback |
  | thread_id | LangGraph conversation thread identifier |
  | topic_filter | Current topic restriction applied to retrieval |
  | difficulty_filter | Current difficulty restriction applied to retrieval |

- **Stretch features implemented:**
  - Offline-safe fallback embeddings when the sentence-transformer model cannot download
  - Offline-safe fallback answer and interview-question generation when no LLM provider is configured

---

## Design Decisions

Documented below are the most important deliberate design choices we expect to
defend during the final walkthrough.

1. **Decision: Use Streamlit instead of Gradio**
   **Rationale:**
   The starter code and repo instructions are already structured around
   `src/rag_agent/ui/app.py`, `st.session_state`, and cached shared resources.
   Streamlit minimizes integration work and lets a two-person team focus on
   required functionality instead of UI framework overhead.
   **Interview answer:**
   We chose Streamlit because the starter project already exposes a Streamlit
   application boundary, which reduced implementation risk under deadline. The
   decision let us spend more time on retrieval quality, duplicate detection,
   and citations instead of framework conversion.

2. **Decision: Use local embeddings with `all-MiniLM-L6-v2`**
   **Rationale:**
   Local embeddings avoid external API cost and keep ingestion reliable even if
   a hosted embedding service is unavailable. The model is small enough to load
   quickly on a laptop while still producing useful semantic retrieval for a
   medium-sized course corpus.
   **Interview answer:**
   We used a local MiniLM sentence-transformer because it gave us the best
   speed-to-quality tradeoff for a classroom environment. It also kept the
   corpus local, which simplified setup and removed a billing dependency.

3. **Decision: Use 512-character chunks with 50-character overlap**
   **Rationale:**
   Smaller chunks improve retrieval precision, while overlap preserves concepts
   that span chunk boundaries. Header-aware splitting for Markdown further
   improves semantic coherence by respecting section structure before fallback
   splitting.
   **Interview answer:**
   We aimed for chunks that each explain one interview-relevant idea rather than
   full pages of content. The overlap prevented boundary loss, and the chunk
   size was small enough to keep retrieval precise without making the context
   too fragmented.

4. **Decision: Apply a similarity threshold of 0.15 with retrieval_k = 4**
   **Rationale:**
   The final build uses a hybrid scorer that combines vector similarity, lexical
   overlap, and topic-aware boosting. In this setup, a 0.15 threshold produced a
   better balance between recall for real topic queries and rejection of
   off-topic prompts.
   **Interview answer:**
   We treated retrieval as a trust boundary, not just a convenience. After
   calibration on our authored corpus, 0.15 plus top-4 reranked results gave us
   grounded answers for real deep-learning questions while still letting the
   no-context guard reject unrelated prompts.

---

## QA Test Results

All 11 unit tests passed via `uv run pytest tests/test_vectorstore.py -v`
(11 passed, 1 warning, 14.85s). Integration test results are recorded below.

| Test | Expected | Actual | Pass / Fail |
|---|---|---|---|
| Normal query | Relevant chunks, source cited | `Explain the vanishing gradient problem in LSTMs` returned grounded context with 3 sources | Pass |
| Off-topic query | No context found message | `What is the capital of France?` triggered `no_context_found=True` with 0 sources | Pass |
| Duplicate ingestion | Second upload skipped | Re-ingesting the authored corpus produced `0 ingested, 37 skipped, 0 errors` | Pass |
| Empty query | Graceful error, no crash | `store.query('')` returned no results and the UI blocks blank chat submission | Pass |
| Cross-topic query | Multi-topic retrieval | `How do LSTMs improve on RNNs for Seq2Seq tasks?` returned 4 chunks from `LSTM` and `Seq2Seq` | Pass |

**Unit test breakdown (all passing):**
- TestChunkIdGeneration: 4/4 — deterministic hashing, uniqueness, length
- TestDuplicateDetection: 3/3 — new chunk detection, duplicate detection, skip on ingest
- TestRetrieval: 4/4 — relevant query, irrelevant query, topic filter, score ordering

**Critical failures fixed before submission:**
- LangChain text splitter import path updated to match installed package versions
- Deterministic fallback embeddings added after discovering Python hash randomization caused unstable retrieval
- Hybrid reranking and stricter no-context guard added to stop off-topic false positives
- Stale session state causing LSTM query to return no context — fixed with backend reload and cache reset helper

---

## Known Limitations

- PDF chunking can include noisy headers, footers, equations, and references unless cleaned after ingestion.
- The similarity threshold is manually selected and should ideally be calibrated empirically with more test queries.
- Conversation memory is session-scoped and depends on the app process staying alive.
- A small local embedding model is efficient but may miss subtle semantic distinctions that a larger embedding model would capture.
- The quality of generated interview questions depends directly on corpus quality and chunk specificity.
- Bonus topics (SOM, Boltzmann Machines, GAN) are not yet ingested; the corpus covers all six required core topics.

---

## What We Would Do With More Time

- Add hybrid retrieval that combines vector search with keyword search for edge cases.
- Add a reranking step so initially retrieved chunks can be reordered by a stronger relevance model.
- Improve PDF cleanup to remove bibliography sections and page artifacts automatically.
- Add streaming token output in the chat panel for a more polished demo experience.
- Expand the corpus to cover bonus topics such as GANs, SOMs, and Boltzmann Machines.
- Tune the similarity threshold empirically using a held-out evaluation set rather than manual calibration.

---

## Hour 3 Interview Questions

**Question 1:**
Explain how the three gates in an LSTM work together to address the vanishing
gradient problem.

Model answer:
An LSTM introduces gating mechanisms that control information flow through the
cell state. The forget gate decides what prior information should be discarded,
the input gate decides what new information should be written, and the output
gate controls what information becomes visible in the hidden state. This gated
path makes it easier for important information to persist across long sequences,
which helps mitigate the vanishing gradient problem that affects standard RNNs.

**Question 2:**
How does a Seq2Seq model improve on a basic RNN for sequence transformation
tasks, and where does LSTM memory help in that design?

Model answer:
A basic RNN processes sequences step by step but struggles to preserve long-term
dependencies, especially when the input and output sequences are different in
length or structure. A Seq2Seq model separates encoding and decoding so one
network compresses the source sequence into a learned representation and another
generates the output sequence. LSTM-style gating helps preserve important
context over longer spans, improving the quality of sequence transformations.

**Question 3:**
Why did the team choose content-hash-based duplicate detection instead of using
only filenames when ingesting documents?

Model answer:
Filename-based duplicate detection is brittle because users can rename a file
and upload the same content again without changing its meaning. A content hash
is deterministic for the actual chunk text, so identical content maps to the
same ID even when filenames change. This makes ingestion safer, prevents silent
vector-store duplication, and gives a cleaner demo of retrieval quality.

---

## Team Retrospective

**What clicked:**
- The LangGraph node structure made it easy to isolate and debug each stage of the pipeline independently
- Content-hash-based duplicate detection worked reliably and was straightforward to explain in the demo
- Streamlit session_state kept UI wiring clean once the pattern was understood
- Groq's low-latency API made live demo responses feel responsive and natural

**What confused us:**
- The similarity threshold required trial and error — the right value was not obvious from theory alone
- LangChain deprecation warnings created noise that made it harder to spot real errors during development
- The first LSTM query returning no context revealed that stale session state can silently break retrieval even when the corpus is correct

**One thing each team member would study before a real interview:**
- Corpus Architect + UX Lead (Asha Sri Sayana): Better strategies for PDF cleanup and corpus authoring at scale
- Pipeline Engineer + QA Lead (Rahul Pogula): More rigorous threshold tuning, evaluation metrics, and production monitoring for RAG systems
- Shared Prompt Engineer role: Stronger structured-output reliability and prompt evaluation techniques
