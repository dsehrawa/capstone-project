from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
import requests
from chromadb.utils import embedding_functions


@dataclass
class RetrievalResult:
    answer: str
    contexts: list[dict[str, Any]]


class ContractRetriever:
    def __init__(self, clauses: list[dict[str, Any]], db_dir: str | Path, collection_name: str = "contract_clauses"):
        self.clauses = clauses
        self.texts = [clause["clause_text"] for clause in clauses]
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.db_dir))
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
        self._index_clauses()

    def _index_clauses(self) -> None:
        existing = self.collection.get()
        existing_ids = existing.get("ids", []) if existing else []
        if existing_ids:
            self.collection.delete(ids=existing_ids)
        if not self.texts:
            return
        ids = [f"clause-{idx}" for idx in range(len(self.texts))]
        metadatas = [{"category": self.clauses[idx].get("category", "General Provisions")} for idx in range(len(self.texts))]
        self.collection.add(ids=ids, documents=self.texts, metadatas=metadatas)

    def retrieve(self, question: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self.texts:
            return []
        query_result = self.collection.query(query_texts=[question], n_results=min(top_k, len(self.texts)))
        documents = query_result.get("documents", [[]])[0]
        metadatas = query_result.get("metadatas", [[]])[0]
        distances = query_result.get("distances", [[]])[0]
        results = []
        for idx, clause_text in enumerate(documents):
            score = 1 - float(distances[idx]) if idx < len(distances) else 0.0
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            results.append({"score": score, "clause_text": clause_text, "category": metadata.get("category", "General Provisions")})
        return results


OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"


def ollama_available(timeout: int = 3) -> bool:
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=timeout)
        response.raise_for_status()
        return True
    except requests.RequestException:
        return False


def ollama_chat(model: str, prompt: str, timeout: int = 120) -> str:
    response = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("message", {}).get("content", "").strip()


def build_contract_qa_prompt(question: str, contexts: list[dict[str, Any]], risk_summary: dict[str, Any]) -> str:
    return (
        "You are reviewing a legal contract and answering a user question.\n\n"
        "Instructions:\n"
        "1. Answer only from the retrieved clauses and the provided risk summary.\n"
        "2. If the evidence is weak, say that clearly.\n"
        "3. Keep the answer structured and concise.\n"
        "4. Mention the most relevant clause evidence in plain language.\n\n"
        f"Question: {question}\n\n"
        f"Risk summary: {risk_summary}\n\n"
        "Retrieved clauses:\n"
        + "\n\n".join(f"[{item['category']}] {item['clause_text']}" for item in contexts[:5])
    )


def answer_question(question: str, retriever: ContractRetriever, risk_summary: dict[str, Any]) -> RetrievalResult:
    contexts = retriever.retrieve(question)
    if not contexts:
        return RetrievalResult(answer="No relevant clauses were retrieved.", contexts=contexts)

    if not ollama_available():
        answer = "Top supporting clauses:\n\n" + "\n\n".join(f"- [{item['category']}] {item['clause_text'][:350]}" for item in contexts[:3]) if contexts else "No relevant clauses were retrieved."
        return RetrievalResult(answer=answer, contexts=contexts)

    prompt = build_contract_qa_prompt(question, contexts, risk_summary)
    answer_text = ollama_chat("mistral", prompt, timeout=120)
    return RetrievalResult(answer=answer_text, contexts=contexts)
