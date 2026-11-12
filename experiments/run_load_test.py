
import argparse
import asyncio
import math
import time

from datetime import datetime, timezone
from statistics import mean
from typing import Any

import httpx

from app.providers.standin_client import StandInClient


BASE_URL = "http://127.0.0.1:8000"


SCENARIOS = {
    "steady": {
        "requests": 12,
        "workers": 3,
        "queue_capacity": 8,
        "interval_seconds": 0.5,
        "config": {
            "max_concurrent_requests": 8,
            "overload_slowdown_ms": 500,
            "requests_per_minute": 0,
            "tokens_per_minute": 0,
            "daily_request_budget": 0,
        },
    },
    "burst": {
        "requests": 24,
        "workers": 12,
        "queue_capacity": 24,
        "interval_seconds": 0,
        "config": {
            "max_concurrent_requests": 2,
            "overload_slowdown_ms": 500,
            "requests_per_minute": 0,
            "tokens_per_minute": 0,
            "daily_request_budget": 0,
        },
    },
    "sustained_overload": {
        "requests": 30,
        "workers": 8,
        "queue_capacity": 12,
        "interval_seconds": 0.2,
        "config": {
            "max_concurrent_requests": 2,
            "overload_slowdown_ms": 500,
            "requests_per_minute": 0,
            "tokens_per_minute": 0,
            "daily_request_budget": 0,
        },
    },
    "rpm_halved": {
        "requests": 18,
        "workers": 18,
        "queue_capacity": 18,
        "interval_seconds": 0,
        "config": {
            "max_concurrent_requests": 20,
            "overload_slowdown_ms": 0,
            "requests_per_minute": 15,
            "tokens_per_minute": 0,
            "daily_request_budget": 0,
        },
    },
    "daily_budget_exhausted": {
        "requests": 12,
        "workers": 12,
        "queue_capacity": 12,
        "interval_seconds": 0,
        "config": {
            "max_concurrent_requests": 20,
            "overload_slowdown_ms": 0,
            "requests_per_minute": 0,
            "tokens_per_minute": 0,
            # Set dynamically relative to current daily count.
            "daily_request_budget": 0,
        },
    },
}


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None

    ordered = sorted(values)
    index = max(0, math.ceil(p * len(ordered)) - 1)
    return ordered[index]


def show_percentile(label: str, values: list[float]) -> None:
    print(f"{label} p50: {percentile(values, 0.50)}")
    print(f"{label} p95: {percentile(values, 0.95)}")
    print(f"{label} p99: {percentile(values, 0.99)}")


async def get_service_state(
    client: httpx.AsyncClient,
) -> dict[str, Any]:
    response = await client.get(f"{BASE_URL}/health")
    response.raise_for_status()
    return response.json()


async def get_config(
    client: httpx.AsyncClient,
) -> dict[str, Any]:
    response = await client.get(f"{BASE_URL}/config")
    response.raise_for_status()
    return response.json()


async def put_config(
    client: httpx.AsyncClient,
    config: dict[str, Any],
) -> None:
    response = await client.put(
        f"{BASE_URL}/config",
        json=config,
    )
    response.raise_for_status()


async def run_scenario(name: str) -> None:
    definition = SCENARIOS[name]

    async with httpx.AsyncClient(
        timeout=10.0,
    ) as control_client:

        health = await get_service_state(control_client)

        if health["status"] != "ok":
            raise RuntimeError("Stand-in service is not healthy.")

        if health["active_requests"] != 0:
            raise RuntimeError(
                "There are active requests. Wait for them to finish."
            )

        original_config = await get_config(control_client)
        scenario_config = original_config.copy()
        scenario_config.update(definition["config"])

        if name == "daily_budget_exhausted":
            current_daily = health["daily_requests"]
            scenario_config["daily_request_budget"] = (
                current_daily + 8
            )

        await put_config(control_client, scenario_config)

        print(f"\n========== SCENARIO: {name} ==========")
        print("Requests:", definition["requests"])
        print("Workers:", definition["workers"])
        print("Queue capacity:", definition["queue_capacity"])
        print("Starting daily counter:", health["daily_requests"])
        print(
            "Configured daily budget:",
            scenario_config["daily_request_budget"],
        )

        queue: asyncio.Queue = asyncio.Queue(
            maxsize=definition["queue_capacity"]
        )

        results: list[dict[str, Any]] = []
        peak_queue_size = 0

        worker_count = definition["workers"]
        request_count = definition["requests"]
        interval = definition["interval_seconds"]

        standin_client = StandInClient(
            base_url=BASE_URL,
            max_retries=2,
            deadline_seconds=30.0,
        )

        async def worker(
            worker_number: int,
            request_client: httpx.AsyncClient,
        ) -> None:
            while True:
                item = await queue.get()

                try:
                    if item is None:
                        return

                    request_number, enqueued_at = item
                    worker_started = time.perf_counter()

                    queue_wait_ms = (
                        worker_started - enqueued_at
                    ) * 1000

                    result = await standin_client.generate(
                        request_client,
                        scenario=name,
                        messages=[
                            {
                                "role": "user",
                                "content": (
                                    f"Load-test request {request_number}. "
                                    "Explain the purpose of a RAG system."
                                ),
                            }
                        ],
                    )

                    result["queue_wait_ms"] = round(
                        queue_wait_ms,
                        2,
                    )
                    result["worker"] = worker_number

                    results.append(result)

                finally:
                    queue.task_done()

        async def producer() -> None:
            nonlocal peak_queue_size

            for request_number in range(1, request_count + 1):
                if interval > 0 and request_number > 1:
                    await asyncio.sleep(interval)

                await queue.put(
                    (request_number, time.perf_counter())
                )

                peak_queue_size = max(
                    peak_queue_size,
                    queue.qsize(),
                )

            for _ in range(worker_count):
                await queue.put(None)

        limits = httpx.Limits(
            max_connections=worker_count,
            max_keepalive_connections=worker_count,
        )

        try:
            async with httpx.AsyncClient(
                limits=limits,
                timeout=35.0,
            ) as request_client:

                workers = [
                    asyncio.create_task(
                        worker(i + 1, request_client)
                    )
                    for i in range(worker_count)
                ]

                experiment_start = time.perf_counter()

                await producer()
                await queue.join()
                await asyncio.gather(*workers)

                experiment_duration = (
                    time.perf_counter() - experiment_start
                )

        finally:
            # Always restore the exact original service configuration.
            await put_config(control_client, original_config)

        statuses = [
            result["status"] for result in results
        ]

        latencies = [
            result["total_ms"] for result in results
        ]

        ttfts = [
            result["first_token_ms"]
            for result in results
            if result["first_token_ms"] is not None
        ]

        queue_waits = [
            result["queue_wait_ms"] for result in results
        ]

        success_count = statuses.count("ok")
        rate_limited_count = statuses.count("rate_limited")
        timeout_count = statuses.count("timeout")
        error_count = statuses.count("error")

        total = len(results)

        input_tokens = [
            result["tokens_in"]
            for result in results
            if result["tokens_in"] is not None
        ]

        output_tokens = [
            result["tokens_out"]
            for result in results
            if result["tokens_out"] is not None
        ]

        retries = sum(
            result["retries"] for result in results
        )

        print("\n========== RESULTS ==========")
        print("Logical requests completed:", total)
        print("Experiment duration (seconds):", round(
            experiment_duration, 2
        ))
        print("Nominal offered rate (req/sec):", (
            round(1 / interval, 2) if interval > 0 else "burst"
        ))
        print("Observed completion throughput (req/sec):", round(
            total / experiment_duration, 2
        ))
        print("Peak queue length:", peak_queue_size)

        print("\nOutcomes:")
        print(
            f"Success: {success_count} "
            f"({success_count / total * 100:.1f}%)"
        )
        print(
            f"Rate limited: {rate_limited_count} "
            f"({rate_limited_count / total * 100:.1f}%)"
        )
        print(
            f"Timeout: {timeout_count} "
            f"({timeout_count / total * 100:.1f}%)"
        )
        print(
            f"Error/given up: {error_count} "
            f"({error_count / total * 100:.1f}%)"
        )
        print("Total retries:", retries)
        print(
            "Average retries/request:",
            round(retries / total, 3),
        )

        print("\nLatency:")
        show_percentile("TTFT ms", ttfts)
        show_percentile("Total ms", latencies)

        print(
            "Average queue wait ms:",
            round(mean(queue_waits), 2) if queue_waits else None,
        )

        print("\nTokens:")
        print(
            "Average input tokens:",
            round(mean(input_tokens), 2) if input_tokens else None,
        )
        print(
            "Average output tokens:",
            round(mean(output_tokens), 2) if output_tokens else None,
        )

        print("\nCSV and trace rows written:", total)

        final_health = await get_service_state(control_client)
        print(
            "Active requests after test:",
            final_health["active_requests"],
        )
        print(
            "Daily requests after test:",
            final_health["daily_requests"],
        )

        if final_health["active_requests"] != 0:
            print("WARNING: Service still reports active requests.")


async def main(name: str) -> None:
    await run_scenario(name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run one controlled stand-in workload scenario."
    )

    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()),
        default="steady",
    )

    args = parser.parse_args()

    asyncio.run(main(args.scenario))
