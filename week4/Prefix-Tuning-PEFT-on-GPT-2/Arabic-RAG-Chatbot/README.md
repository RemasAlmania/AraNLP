
# Arabic RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions in Arabic using Arabic Wikipedia as its knowledge base.

---

## Project Requirements

- Arabic dataset with 30,000+ samples
- Three text splitting methods: Fixed, Recursive, Semantic
- Arabic-compatible embedding model
- LLM that supports Arabic
- RAG chatbot with Gradio interface

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Dataset | Arabic Wikipedia via wikimedia/wikipedia (34,000+ articles) |
| Embedding Model | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 |
| Vector Store | FAISS |
| LLM | OpenAI GPT-4o-mini |
| UI | Gradio |

---

## Text Splitting Methods

### 1. Fixed Size Chunking
Splits text into chunks of fixed character size (500 chars) with overlap (50 chars), regardless of sentence boundaries.

### 2. Recursive Character Splitting
Tries to split by paragraphs, then sentences, then words, then characters — preserving natural text structure. Includes Arabic punctuation separators.

### 3. Semantic Chunking
Uses sentence embeddings and cosine similarity to detect topic shifts between sentences. Groups semantically similar sentences into the same chunk.

---

## How It Works

1. Load 35,000 Arabic Wikipedia articles
2. Split articles using all three methods
3. Build FAISS vector index from recursive chunks
4. User submits a question in Arabic
5. System retrieves top 5 most relevant chunks via similarity search
6. GPT-4o-mini generates an answer in Arabic based only on retrieved context

---

## Project Structure

```
Arabic_RAG_Chatbot.ipynb   — main notebook (7 sections)
README.md                  — project documentation
```

---

## How to Run

1. Open `Arabic_RAG_Chatbot.ipynb` in Google Colab
2. Run Section 1 to install all libraries
3. Add your OpenAI API key in Section 2
4. Run all sections in order from top to bottom
5. The Gradio interface will launch at the end of Section 7

---

## Dataset

- Source: [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia)
- Language: Arabic (20231101.ar)
- Size: 34,000+ articles after filtering

---

## Example Questions 

- ما هي الرياضيات وما فروعها الرئيسية؟
- من هو ابن خلدون وما أبرز مؤلفاته؟
- أين تقع إستونيا وما عاصمتها؟
- ما هي مدينة عمان وما تاريخها؟
## Run Sample 

<img width="2527" height="1651" alt="ADEB87EF-A4EE-4D2B-AE46-310AF10CE0E2_1_201_a" src="https://github.com/user-attachments/assets/3801bc92-6c0c-4426-b049-475462d328d0" />
