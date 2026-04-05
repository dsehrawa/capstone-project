# Contract Risk Intelligence App

This folder contains a Streamlit app for the next deliverable:

- upload a contract PDF
- run extraction, risk analysis, obligation detection, and live external-risk enrichment
- view a structured result
- ask grounded questions over the uploaded contract

## Run

```powershell
pip install -r requirements.txt
streamlit run app.py
```

## Features

- **Contract Analysis**: Upload PDF, extract text, classify clauses, detect risks/obligations, enrich with external APIs.
- **Q&A**: Ask grounded questions over the contract using RAG (vector search + LLM).
- **Golden Dataset Testing**: Run 40 baseline questions to evaluate RAG performance, with metrics and detailed comparisons.
- **Exports**: Download reports as CSV/JSON.

## Notes

- Live external risks use NOAA, CISA KEV, and World Bank APIs.
- Clause embeddings are stored locally in a Chroma vector database under `chroma_db/`.
- Q&A uses local Ollama with `mistral` if available, and falls back to extractive retrieval if Ollama is not running.
- For better entity extraction, install the SpaCy English model:

```powershell
python -m spacy download en_core_web_sm
```

- Start Ollama before using full Q&A:

```powershell
ollama serve
ollama pull mistral
```

## Testing

- Use `test_rag.py` to run golden dataset tests programmatically.
- The UI includes a "Run Golden Tests" button for interactive evaluation.
