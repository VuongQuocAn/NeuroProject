from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


TUMOR_ALIASES = {
    "glioma": "glioma",
    "glioblastoma": "glioma",
    "u than kinh dem": "glioma",
    "meningioma": "meningioma",
    "u mang nao": "meningioma",
    "pituitary": "pituitary",
    "pituitary tumor": "pituitary",
    "pituitary_tumor": "pituitary",
    "u tuyen yen": "pituitary",
}


@dataclass
class RetrievedContext:
    child_id: str
    parent_id: str
    score: float
    child_text: str
    parent_text: str
    source_title: str | None
    source_url: str | None
    labels: dict[str, Any]


class LocalXaiRagService:
    """Local child-parent RAG over precomputed BGE-M3 child embeddings."""

    def __init__(self, store_dir: str | Path | None = None):
        backend_dir = Path(__file__).resolve().parents[1]
        self.store_dir = Path(store_dir) if store_dir else backend_dir / "rag" / "xai"
        self.config_path = self.store_dir / "embedding_config.json"
        self.embeddings_path = self.store_dir / "child_embeddings.npy"
        self.children_path = self.store_dir / "child_chunks_metadata.jsonl"
        self.parents_path = self.store_dir / "parent_chunks_lookup.json"

        self.config: dict[str, Any] = {}
        self.children: list[dict[str, Any]] = []
        self.parents: dict[str, dict[str, Any]] = {}
        self.embeddings: np.ndarray | None = None
        self._local_model = None

        self._load_store()

    def _load_store(self) -> None:
        missing = [
            str(path)
            for path in (self.config_path, self.embeddings_path, self.children_path, self.parents_path)
            if not path.exists()
        ]
        if missing:
            raise RuntimeError(f"Missing local XAI RAG files: {missing}")

        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.embeddings = np.load(self.embeddings_path).astype(np.float32)

        with self.children_path.open("r", encoding="utf-8") as file:
            self.children = [json.loads(line) for line in file if line.strip()]

        self.parents = json.loads(self.parents_path.read_text(encoding="utf-8"))

        if self.embeddings.shape[0] != len(self.children):
            raise RuntimeError(
                "child_embeddings.npy and child_chunks_metadata.jsonl length mismatch: "
                f"{self.embeddings.shape[0]} vs {len(self.children)}"
            )

        self.embeddings = self._l2_normalize(self.embeddings)

    def retrieve_classification_context(
        self,
        tumor_label: str | None,
        classification_confidence: float | None = None,
        xai_metadata: dict[str, Any] | None = None,
        top_k: int = 3,
        candidate_k: int = 10,
    ) -> tuple[list[RetrievedContext], dict[str, Any]]:
        if self.embeddings is None:
            raise RuntimeError("RAG embeddings are not loaded.")

        tumor_key = self._normalize_tumor(tumor_label)
        query_text = self._build_classification_query(tumor_key, tumor_label, classification_confidence, xai_metadata)
        query_embedding = self._embed_query(query_text)

        candidate_indices = self._filter_child_indices(tumor_key=tumor_key)
        if not candidate_indices:
            candidate_indices = self._filter_child_indices(tumor_key=None)
        if not candidate_indices:
            candidate_indices = list(range(len(self.children)))

        candidate_vectors = self.embeddings[candidate_indices]
        scores = candidate_vectors @ query_embedding
        order = np.argsort(scores)[::-1][: max(candidate_k, top_k)]

        dense_contexts: list[RetrievedContext] = []
        seen_parent_ids: set[str] = set()

        for local_idx in order:
            child_idx = int(candidate_indices[int(local_idx)])
            child = self.children[child_idx]
            parent_id = str(child.get("parent_id") or "")
            parent = self.parents.get(parent_id, {})

            parent_text = str(parent.get("text") or "")
            child_text = str(child.get("text") or "")
            if not parent_text and not child_text:
                continue

            dedupe_key = parent_id or str(child.get("child_id") or child.get("record_id") or child_idx)
            if dedupe_key in seen_parent_ids:
                continue
            seen_parent_ids.add(dedupe_key)

            dense_contexts.append(
                RetrievedContext(
                    child_id=str(child.get("child_id") or child.get("record_id") or child_idx),
                    parent_id=parent_id,
                    score=float(scores[int(local_idx)]),
                    child_text=child_text,
                    parent_text=parent_text,
                    source_title=str(parent.get("page_title") or child.get("page_title") or "") or None,
                    source_url=str(parent.get("source_url") or child.get("source_url") or "") or None,
                    labels=dict(child.get("xai_rag_labels") or {}),
                )
            )

            if len(dense_contexts) >= max(candidate_k, top_k):
                break

        top_contexts, rerank_info = self._rerank_contexts(
            query_text=query_text,
            contexts=dense_contexts,
            top_k=top_k,
        )

        diagnostics = {
            "query_text": query_text,
            "tumor_filter": tumor_key,
            "candidate_count": len(candidate_indices),
            "dense_candidate_count": len(dense_contexts),
            "top_k": len(top_contexts),
            "embedding_model": self.config.get("model_name", "BAAI/bge-m3"),
            "rerank": rerank_info,
            "store_has_mojibake": self._store_has_mojibake(),
        }
        return top_contexts, diagnostics

    def _filter_child_indices(self, tumor_key: str | None) -> list[int]:
        indices: list[int] = []
        for idx, child in enumerate(self.children):
            labels = child.get("xai_rag_labels") or {}
            tasks = set(labels.get("task") or [])
            stages = set(labels.get("pipeline_stage") or [])
            tumors = {self._normalize_tumor(value) for value in labels.get("tumor") or []}

            if "explain_classification" not in tasks and "classification" not in stages:
                continue
            if tumor_key and tumor_key not in tumors:
                continue
            indices.append(idx)
        return indices

    def _embed_query(self, text: str) -> np.ndarray:
        token = os.getenv("HF_API_TOKEN") or os.getenv("HF_TOKEN")
        if not token:
            raise RuntimeError(
                "Missing HF_API_TOKEN. Query embedding is configured to use Hugging Face API only "
                "because BAAI/bge-m3 is too heavy for local runtime."
            )

        return self._embed_query_with_hf_api(text=text, token=token)

    def _embed_query_with_hf_api(self, text: str, token: str) -> np.ndarray:
        import requests

        model_name = self.config.get("model_name") or "BAAI/bge-m3"
        url = (
            os.getenv("HF_EMBEDDING_API_URL")
            or f"https://router.huggingface.co/hf-inference/models/{model_name}/pipeline/feature-extraction"
        )
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {token}"},
            json={"inputs": text, "options": {"wait_for_model": True}},
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        vector = np.asarray(payload, dtype=np.float32)
        if vector.ndim == 2:
            vector = vector.mean(axis=0)
        if vector.ndim == 3:
            vector = vector[0].mean(axis=0)
        if vector.ndim != 1:
            raise RuntimeError(f"Unexpected HF embedding shape: {vector.shape}")
        return self._l2_normalize(vector)

    def _rerank_contexts(
        self,
        query_text: str,
        contexts: list[RetrievedContext],
        top_k: int,
    ) -> tuple[list[RetrievedContext], dict[str, Any]]:
        if not contexts:
            return [], {"applied": False, "reason": "no_dense_contexts"}

        token = os.getenv("HF_API_TOKEN") or os.getenv("HF_TOKEN")
        model_name = os.getenv("HF_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
        url = os.getenv("HF_RERANKER_API_URL") or f"https://router.huggingface.co/hf-inference/models/{model_name}"

        if not token:
            return contexts[:top_k], {
                "applied": False,
                "reason": "missing_hf_token",
                "fallback": "dense_order",
            }

        try:
            import requests

            pairs = [
                {
                    "text": query_text,
                    "text_pair": (context.child_text or context.parent_text or "")[:4000],
                }
                for context in contexts
            ]
            response = requests.post(
                url,
                headers={"Authorization": f"Bearer {token}"},
                json={"inputs": pairs, "options": {"wait_for_model": True}},
                timeout=120,
            )
            response.raise_for_status()
            rerank_scores = self._parse_reranker_scores(response.json(), expected=len(contexts))
            if len(rerank_scores) != len(contexts):
                raise RuntimeError(f"Unexpected reranker score count: {len(rerank_scores)} vs {len(contexts)}")

            reranked: list[RetrievedContext] = []
            for context, rerank_score in zip(contexts, rerank_scores):
                reranked.append(
                    RetrievedContext(
                        child_id=context.child_id,
                        parent_id=context.parent_id,
                        score=float(rerank_score),
                        child_text=context.child_text,
                        parent_text=context.parent_text,
                        source_title=context.source_title,
                        source_url=context.source_url,
                        labels=context.labels,
                    )
                )

            reranked.sort(key=lambda item: item.score, reverse=True)
            return reranked[:top_k], {
                "applied": True,
                "model": model_name,
                "candidate_count": len(contexts),
                "top_k": top_k,
                "score_type": "cross_encoder_relevance",
            }
        except Exception as exc:
            return contexts[:top_k], {
                "applied": False,
                "reason": str(exc),
                "fallback": "dense_order",
                "model": model_name,
            }

    def _parse_reranker_scores(self, payload: Any, expected: int) -> list[float]:
        if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], list):
            payload = payload[0]

        scores: list[float] = []
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, list) and item and isinstance(item[0], dict):
                    scores.append(float(item[0].get("score", 0.0)))
                elif isinstance(item, dict):
                    scores.append(float(item.get("score", 0.0)))

        if len(scores) == expected:
            return scores
        return scores

    def _build_classification_query(
        self,
        tumor_key: str | None,
        tumor_label: str | None,
        classification_confidence: float | None,
        xai_metadata: dict[str, Any] | None,
    ) -> str:
        tumor_vi = {
            "glioma": "u than kinh dem",
            "meningioma": "u mang nao",
            "pituitary": "u tuyen yen",
        }.get(tumor_key or "", tumor_label or "u nao")

        parts = [
            "Explain an MRI brain tumor classification result in Vietnamese.",
            f"Predicted tumor class: {tumor_label or tumor_vi}.",
            f"Vietnamese tumor name: {tumor_vi}.",
            "Need clinical background for the tumor type and safe explanation of why classification may be plausible.",
        ]
        if classification_confidence is not None:
            parts.append(f"Classification confidence: {classification_confidence:.4f}.")
        if xai_metadata:
            method = xai_metadata.get("method") or xai_metadata.get("classification_method")
            target_class = xai_metadata.get("target_class") or xai_metadata.get("target_idx")
            if method:
                parts.append(f"XAI method metadata: {method}.")
            if target_class is not None:
                parts.append(f"XAI target class metadata: {target_class}.")
        return " ".join(parts)

    def _normalize_tumor(self, value: Any) -> str | None:
        if value is None:
            return None
        raw = str(value).strip().lower().replace("-", " ").replace("_", " ")
        raw = raw.replace("u thần kinh đệm", "u than kinh dem")
        raw = raw.replace("u màng não", "u mang nao")
        raw = raw.replace("u tuyến yên", "u tuyen yen")
        for alias, normalized in TUMOR_ALIASES.items():
            if alias in raw:
                return normalized
        return raw or None

    def _store_has_mojibake(self) -> bool:
        sample_parts = []
        for child in self.children[:20]:
            sample_parts.append(str(child.get("text", ""))[:300])
            labels = child.get("xai_rag_labels") or {}
            sample_parts.append(" ".join(str(v) for v in labels.get("tumor_vi") or []))
        sample = "\n".join(sample_parts)
        return any(marker in sample for marker in ("Ã", "Ä", "á»", "Æ"))

    @staticmethod
    def _l2_normalize(array: np.ndarray) -> np.ndarray:
        arr = np.asarray(array, dtype=np.float32)
        if arr.ndim == 1:
            norm = float(np.linalg.norm(arr))
            return arr / max(norm, 1e-8)
        norm = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norm, 1e-8)


@lru_cache(maxsize=1)
def get_xai_rag_service() -> LocalXaiRagService:
    return LocalXaiRagService()
