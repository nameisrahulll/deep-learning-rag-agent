# 5-Minute Demo Script

## Goal

Show the working RAG interview-prep flow in this order:
1. Corpus ingestion
2. Duplicate detection
3. Normal query with citations
4. Off-topic hallucination guard
5. Interview-question generation

---

## Suggested Talk Track

### 0:00 - 0:30 | Intro

"This is our Deep Learning RAG Interview Prep Agent. It is built with
LangChain, LangGraph, ChromaDB, and Streamlit. The system ingests deep
learning study material, stores it in a vector database, and lets a user
chat with the corpus to answer and generate interview questions."

---

### 0:30 - 1:20 | Show the Corpus

- Open the app sidebar
- Point out the uploaded Markdown corpus files:
  - `ann_intermediate.md`
  - `cnn_intermediate.md`
  - `rnn_intermediate.md`
  - `lstm_intermediate.md`
  - `seq2seq_intermediate.md`
  - `autoencoder_intermediate.md`
- Mention that the current validated corpus contains 37 chunks across 6 core topics

Say:
"Our corpus currently covers ANN, CNN, RNN, LSTM, Seq2Seq, and Autoencoder.
Each chunk carries metadata like topic, difficulty, type, source, related
topics, and whether the topic is bonus content."

---

### 1:20 - 2:00 | Ingestion and Duplicate Detection

- Upload one or more of the existing corpus files
- Click **Ingest Documents**
- Show the success message for chunks added
- Upload the same file(s) again
- Click **Ingest Documents** again
- Show that the second pass skips duplicates

Say:
"Chunk IDs are generated from a content hash of source plus chunk text, so
re-uploading the same material does not duplicate it in ChromaDB."

---

### 2:00 - 3:00 | Normal Query With Sources

- Ask: `How do LSTMs solve the vanishing gradient problem?`
- Show that the answer is returned with source citations
- Open the document viewer and show the corresponding LSTM chunks

Say:
"The query goes through a LangGraph pipeline with query rewriting,
retrieval, and generation. The response is grounded in retrieved chunks and
the source citations are shown in the UI."

---

### 3:00 - 3:40 | Off-Topic Guard

- Ask: `What is the capital of France?`
- Show the no-context response

Say:
"This demonstrates the hallucination guard. If no chunk passes the
relevance threshold, the agent refuses to answer from unsupported context
instead of fabricating a deep-learning response."

---

### 3:40 - 4:30 | Interview Question Generation

- Ask: `Generate an interview question about Seq2Seq models.`
- Show the generated interview question, model answer, and follow-up

Say:
"The same corpus can also be used to generate interview-style questions
from the readings, which is part of the assignment requirement."

---

### 4:30 - 5:00 | Close

- Mention the architecture document
- Mention the passing vector store tests
- Mention the public GitHub repo link in the submission

Say:
"We also documented the system design in `docs/architecture.md` and verified
the vector store behavior with unit tests covering duplicate detection,
retrieval, filtering, and the hallucination guard."
