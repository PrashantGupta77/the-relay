from datetime import datetime, timezone
from time import perf_counter

from app.evaluation.request_logger import log_rag_request
from app.pipeline.orchestrator import RAGOrchestrator
from app.providers.groq_provider import GroqProvider
from app.retrieval.retriever import Retriever


def main():
    question = (
        "How does scaled dot-product attention "
        "calculate attention weights?"
    )

    print("=" * 70)
    print("THE RELAY - FIRST END-TO-END RAG TEST")
    print("=" * 70)

    provider = GroqProvider(timeout=30.0)
    retriever = Retriever(setting="b")

    pipeline = RAGOrchestrator(
        provider=provider,
        retriever=retriever,
        top_k=5,
        max_completion_tokens=512,
    )

    start_time = datetime.now(timezone.utc)
    timer_start = perf_counter()

    try:
        result = pipeline.run(
            question=question,
            paper_id="attention_is_all_you_need",
        )

    except Exception as exc:
        total_ms = (perf_counter() - timer_start) * 1000

        request_id = log_rag_request(
            result=None,
            start_time=start_time,
            total_ms=total_ms,
            error=exc,
        )

        print("\nREQUEST FAILED:", type(exc).__name__)
        print("Error:", str(exc))
        print("Logged request ID:", request_id)
        raise

    total_ms = (perf_counter() - timer_start) * 1000

    request_id = log_rag_request(
        result=result,
        start_time=start_time,
        total_ms=total_ms,
    )

    print("\nLOGGED REQUEST ID:", request_id)

    print("\nSTATUS:", result["status"])
    print("\nANSWER:")
    print(result["answer"])

    print("\nCITATIONS:")
    for citation in result["citations"]:
        print(citation)

    print("\nRETRIEVAL:")
    for item in result["retrieval"]["results"]:
        print(
            item["rank"],
            item["chunk_id"],
            round(item["score"], 4),
        )

    print("\nPROVIDER METRICS:")
    metrics = result["provider"]

    if metrics:
        for key, value in metrics.items():
            if key != "rate_limit_headers":
                print(f"{key}: {value}")

    print("\nEND-TO-END LATENCY:", round(total_ms, 2), "ms")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
