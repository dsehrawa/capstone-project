from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
import requests


@dataclass
class RetrievalResult:
    answer: str
    contexts: list[dict[str, Any]]


class ContractRetriever:
    def __init__(self, clauses: list[dict[str, Any]], db_dir: str | Path, collection_name: str = "contract_clauses"):
        self.clauses = [clause for clause in clauses if clause.get("clause_text")]
        self.texts = [clause["clause_text"] for clause in self.clauses]
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.collection = None
        self._index_clauses()

    def _index_clauses(self) -> None:
        if not self.texts:
            self.collection = None
            return
        client = chromadb.PersistentClient(path=str(self.db_dir))
        # Re-index fresh each time to stay in sync with input clauses.
        try:
            client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.collection.add(
            documents=self.texts,
            ids=[f"clause_{i}" for i in range(len(self.texts))],
            metadatas=[
                {
                    "index": i,
                    "category": self.clauses[i].get("category", "General Provisions"),
                }
                for i in range(len(self.clauses))
            ],
        )

    def _expand_query(self, question: str) -> list[str]:
        """Generate additional query variants to improve retrieval recall."""
        queries = [question]
        if not ollama_available():
            return queries
        prompt = (
            "Generate 3 short keyword-rich search queries to find relevant legal contract clauses "
            "for the following question. Return ONLY the queries, one per line, no numbering.\n\n"
            f"Question: {question}"
        )
        try:
            raw = ollama_chat("mistral", prompt, timeout=30)
            for line in raw.strip().splitlines():
                line = line.strip().lstrip("0123456789.-) ")
                if len(line) > 10:
                    queries.append(line)
        except Exception:
            pass
        return queries[:4]

    def retrieve(self, question: str, top_k: int = 8) -> list[dict[str, Any]]:
        if not self.texts or self.collection is None:
            return []

        queries = self._expand_query(question)
        n_results = min(top_k, len(self.texts))

        score_map: dict[int, float] = {}
        all_hit_indices: list[int] = []

        for q in queries:
            query_result = self.collection.query(
                query_texts=[q],
                n_results=n_results,
            )
            indices = [m["index"] for m in query_result["metadatas"][0]]
            scores = [1.0 - d for d in query_result["distances"][0]]
            all_hit_indices.extend(indices)

            for idx, chroma_score in zip(indices, scores):
                # Keep the best Chroma similarity across all query variants.
                if idx not in score_map or chroma_score > score_map[idx]:
                    score_map[idx] = chroma_score

        # Expand context window: include +/-2 neighbours of top hits.
        expanded: set[int] = set()
        for idx in all_hit_indices:
            for offset in (-2, -1, 0, 1, 2):
                neighbour = idx + offset
                if 0 <= neighbour < len(self.texts):
                    if neighbour not in score_map:
                        # Neighbour clauses are included for context with a low inherited score.
                        score_map[neighbour] = 0.0
                    expanded.add(neighbour)

        expanded_ranked = sorted(expanded, key=lambda i: score_map[i], reverse=True)
        expanded_ranked = expanded_ranked[: min(top_k + 4, len(expanded_ranked))]

        results = []
        for clause_index in expanded_ranked:
            clause = self.clauses[clause_index]
            results.append(
                {
                    "score": float(score_map[clause_index]),
                    "clause_text": clause.get("clause_text", ""),
                    "category": clause.get("category", "General Provisions"),
                }
            )
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
    numbered_clauses = "\n\n".join(
        f"Clause {idx}. [{item['category']}] {item['clause_text']}"
        for idx, item in enumerate(contexts[:6], start=1)
    )
    return (
        "You are an expert legal analyst reviewing a contract.\n\n"
        "TASK: Answer the user's question using ONLY the contract clauses and risk summary provided below.\n\n"
        "RULES:\n"
        "1. Ground every claim in a specific clause. Refer to clauses by their number (e.g. 'Clause 3').\n"
        "2. Quote or closely paraphrase the exact contract language that supports your answer.\n"
        "3. Include specific details such as dates, durations, percentages, party names, and conditions when available.\n"
        "4. If the clauses do not contain enough information, say so explicitly and explain what is missing.\n"
        "5. Keep the answer concise and evidence-based.\n\n"
        f"QUESTION: {question}\n\n"
        "--- RISK SUMMARY ---\n"
        f"{risk_summary}\n\n"
        "--- RETRIEVED CONTRACT CLAUSES ---\n"
        f"{numbered_clauses}\n"
    )


def answer_question(question: str, retriever: ContractRetriever, risk_summary: dict[str, Any]) -> RetrievalResult:
    contexts = retriever.retrieve(question, top_k=8)
    if not contexts:
        return RetrievalResult(answer="No relevant clauses were retrieved.", contexts=contexts)

    if not ollama_available():
        answer = "Top supporting clauses:\n\n" + "\n\n".join(
            f"- [{item['category']}] {item['clause_text'][:350]}" for item in contexts[:5]
        )
        return RetrievalResult(answer=answer, contexts=contexts)

    prompt = build_contract_qa_prompt(question, contexts, risk_summary)
    try:
        answer_text = ollama_chat("mistral", prompt, timeout=300)
        return RetrievalResult(answer=answer_text, contexts=contexts)
    except requests.RequestException:
        fallback = "Top supporting clauses (LLM timed out):\n\n" + "\n\n".join(
            f"- [{item['category']}] {item['clause_text'][:350]}" for item in contexts[:5]
        )
        return RetrievalResult(answer=fallback, contexts=contexts)
