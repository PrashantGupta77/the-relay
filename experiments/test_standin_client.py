
import asyncio
import httpx

from app.providers.standin_client import StandInClient


async def main():
    client = StandInClient(
        base_url="http://127.0.0.1:8000",
        max_retries=2,
        deadline_seconds=30,
    )

    async with httpx.AsyncClient() as http_client:
        result = await client.generate(
            http_client,
            scenario="queue_wait_logging_smoke",
            messages=[
                {
                    "role": "user",
                    "content": "Explain the purpose of a RAG system.",
                }
            ],
            queue_wait_ms=125.0,
        )

    print("\n========== CLIENT RESULT ==========")
    print("Request ID:", result["request_id"])
    print("Status:", result["status"])
    print("TTFT:", result["first_token_ms"])
    print("Total latency:", result["total_ms"])
    print("Retries:", result["retries"])
    print("Input tokens:", result["tokens_in"])
    print("Output tokens:", result["tokens_out"])
    print("Finish reason:", result["finish_reason"])
    print("Response length:", len(result["content"]))
    print("Attempts:", len(result["attempts"]))

    print("\nResponse:")
    print(result["content"])

    assert result["status"] == "ok"
    assert result["first_token_ms"] is not None
    assert result["tokens_in"] is not None
    assert result["tokens_out"] is not None
    assert result["content"]
    assert result["finish_reason"] == "stop"
    assert result["retries"] == 0

    print("\nSMOKE TEST: PASS")


if __name__ == "__main__":
    asyncio.run(main())
