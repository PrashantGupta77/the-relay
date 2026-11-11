import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq


# Project root and environment configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)

API_KEY = os.getenv("GROQ_API_KEY")
SELECTED_MODEL = os.getenv("GROQ_MODEL")

if not API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing from .env")

if not SELECTED_MODEL:
    raise RuntimeError("GROQ_MODEL is missing from .env")


def main():
    print("Connecting to Groq API...")
    print(f"Selected model: {SELECTED_MODEL}")

    client = Groq(
        api_key=API_KEY,
        timeout=20.0,
        max_retries=0,
    )

    print("\nFetching available models...")

    response = client.models.list()

    models = []

    for model in response.data:
        models.append({
            "id": model.id,
            "owned_by": getattr(model, "owned_by", None),
        })

    available_ids = {model["id"] for model in models}

    selected_available = SELECTED_MODEL in available_ids

    output = {
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "selected_model": SELECTED_MODEL,
        "selected_model_available": selected_available,
        "model_count": len(models),
        "models": sorted(models, key=lambda item: item["id"]),
    }

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_file = results_dir / "model_info.json"

    output_file.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print(f"\nModels retrieved: {len(models)}")
    print(f"Selected model available: {selected_available}")

    print("\nChecking selected model:")

    if selected_available:
        print(f"SUCCESS: {SELECTED_MODEL} is available.")
    else:
        print(f"WARNING: {SELECTED_MODEL} was not found.")
        print("Check the model ID and account model permissions.")

    print(f"\nModel information saved to: {output_file}")

    print("\nOther candidate models:")

    candidates = [
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "qwen/qwen3.8-27b",
    ]

    for candidate in candidates:
        status = "AVAILABLE" if candidate in available_ids else "NOT FOUND"
        print(f"{candidate}: {status}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nGroq API connection failed.")
        print(f"Error type: {type(exc).__name__}")
        print(f"Error details: {exc}")
        raise
