import re
from typing import Any

from app.providers.base import BaseProvider
from app.retrieval.retriever import Retriever, format_citation


CITATION_PATTERN = re.compile(r"\[(E\d+)\]")

SYSTEM_PROMPT = """
You are a research assistant answering questions using only supplied
research-paper evidence.

Rules:
1. Answer only using the supplied evidence.
2. Do not use outside knowledge or invent facts.
3. If the evidence does not sufficiently answer the question, respond
   with exactly this prefix:
   INSUFFICIENT_EVIDENCE:
   followed by a brief explanation.
4. Cite every substantive factual claim using evidence IDs.
   Write each citation as a separate bracketed ID, for example [E1][E2].
5. Never combine multiple evidence IDs inside one pair of brackets.
   Do not write [E1, E2] or [E1, E2, E3].
6. Only use evidence IDs supplied in the prompt.
7. Preserve important technical details, equations and distinctions.
8. Do not invent paper titles, section numbers or page numbers.
9. Do not claim that a statement appears in a paper unless the evidence
   supports it.
10. Be concise but sufficiently detailed.
"""


class RAGOrchestrator:
    """Connect retrieval, generation and citation-integrity validation."""

    def __init__(
        self,
        provider: BaseProvider,
        retriever: Retriever | None = None,
        setting: str = "b",
        top_k: int = 5,
        max_completion_tokens: int = 512,
    ):
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if max_completion_tokens < 1:
            raise ValueError("max_completion_tokens must be positive.")

        self.provider = provider
        self.retriever = retriever or Retriever(setting=setting)
        self.top_k = top_k
        self.max_completion_tokens = max_completion_tokens

    def _retrieve(
        self,
        question: str,
        paper_id: str | None,
        paper_ids: list[str] | None,
    ) -> dict[str, Any]:
        if paper_id is not None and paper_ids is not None:
            raise ValueError("Use paper_id or paper_ids, not both.")

        if paper_ids is None:
            return self.retriever.retrieve(
                question=question,
                top_k=self.top_k,
                paper_id=paper_id,
            )

        if not paper_ids:
            raise ValueError("paper_ids cannot be empty.")

        combined_results = []

        for current_paper_id in paper_ids:
            result = self.retriever.retrieve(
                question=question,
                top_k=self.top_k,
                paper_id=current_paper_id,
            )
            combined_results.extend(result["results"])

        combined_results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        # Avoid duplicate original chunks.
        unique_results = []
        seen_chunks = set()

        for result in combined_results:
            if result["chunk_id"] in seen_chunks:
                continue

            seen_chunks.add(result["chunk_id"])
            unique_results.append(result)

        unique_results = unique_results[:self.top_k]

        for rank, result in enumerate(unique_results, start=1):
            result["rank"] = rank

        return {
            "question": question,
            "setting": self.retriever.setting,
            "top_k": self.top_k,
            "paper_filter": paper_ids,
            "result_count": len(unique_results),
            "results": unique_results,
        }

    @staticmethod
    def _build_evidence(results: list[dict]) -> tuple[str, dict]:
        evidence_blocks = []
        evidence_lookup = {}

        for index, result in enumerate(results, start=1):
            evidence_id = f"E{index}"

            evidence_lookup[evidence_id] = result

            evidence_blocks.append(
                f"""
[{evidence_id}]
Paper: {result["paper_title"]}
arXiv: {result["arxiv_id"]}{result["arxiv_version"]}
Section: {result["section_path"]}
Pages: {result["page_numbers"]}
Evidence:
{result["text"]}
"""
            )

        return "\n".join(evidence_blocks), evidence_lookup

    @staticmethod
    def _validate_citations(
        answer: str,
        evidence_lookup: dict,
    ) -> tuple[bool, list[str], str | None]:
        citation_ids = list(dict.fromkeys(
            CITATION_PATTERN.findall(answer)
        ))

        if not citation_ids:
            return False, [], "No evidence citation was provided."

        invalid_ids = [
            evidence_id
            for evidence_id in citation_ids
            if evidence_id not in evidence_lookup
        ]

        if invalid_ids:
            return (
                False,
                citation_ids,
                f"Unknown evidence IDs: {invalid_ids}",
            )

        return True, citation_ids, None

    def run(
        self,
        question: str,
        paper_id: str | None = None,
        paper_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        question = question.strip()

        retrieval = self._retrieve(
            question=question,
            paper_id=paper_id,
            paper_ids=paper_ids,
        )

        results = retrieval["results"]

        if not results:
            return {
                "status": "refused",
                "answer": (
                    "I cannot answer this question because no relevant "
                    "evidence was retrieved from the selected papers."
                ),
                "reason": "No evidence retrieved.",
                "citations": [],
                "retrieval": retrieval,
                "provider": None,
            }

        evidence_text, evidence_lookup = self._build_evidence(results)

        user_prompt = f"""
Question:
{question}

Retrieved paper evidence:
{evidence_text}

Answer the question using only the evidence above.
Include valid evidence IDs as separate bracketed markers, for example [E1][E2].
Never combine IDs with commas inside a single pair of brackets.
If the evidence is insufficient, begin with INSUFFICIENT_EVIDENCE:
"""

        response = self.provider.generate(
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            max_completion_tokens=self.max_completion_tokens,
            temperature=0.0,
        )

        answer = (response.text or "").strip()

        provider_metrics = {
            "provider": self.provider.provider_name,
            "model": response.model,
            "latency_ms": response.latency_ms,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "total_tokens": response.total_tokens,
            "finish_reason": response.finish_reason,
            "provider_request_id": response.provider_request_id,
            "rate_limit_headers": response.rate_limit_headers,
        }

        if not answer:
            return {
                "status": "refused",
                "answer": "The model returned an empty answer.",
                "reason": "Empty model response.",
                "citations": [],
                "retrieval": retrieval,
                "provider": provider_metrics,
            }

        if response.finish_reason == "length":
            return {
                "status": "refused",
                "answer": (
                    "The model response was truncated before completion. "
                    "Please retry with a suitable completion-token limit."
                ),
                "reason": "Incomplete model response.",
                "citations": [],
                "retrieval": retrieval,
                "provider": provider_metrics,
            }

        if answer.startswith("INSUFFICIENT_EVIDENCE:"):
            return {
                "status": "refused",
                "answer": answer,
                "reason": "Model reported insufficient evidence.",
                "citations": [],
                "retrieval": retrieval,
                "provider": provider_metrics,
            }

        valid, citation_ids, error = self._validate_citations(
            answer,
            evidence_lookup,
        )

        if not valid:
            return {
                "status": "refused",
                "answer": (
                    "The generated response could not pass citation "
                    "integrity validation, so it has not been presented "
                    "as a supported answer."
                ),
                "reason": error,
                "citations": [],
                "retrieval": retrieval,
                "provider": provider_metrics,
            }

        citations = []

        for evidence_id in citation_ids:
            result = evidence_lookup[evidence_id]

            citations.append({
                "evidence_id": evidence_id,
                "chunk_id": result["chunk_id"],
                "paper_id": result["paper_id"],
                "citation": format_citation(result),
            })

        return {
            "status": "answered",
            "answer": answer,
            "reason": None,
            "citations": citations,
            "retrieval": retrieval,
            "provider": provider_metrics,
        }
