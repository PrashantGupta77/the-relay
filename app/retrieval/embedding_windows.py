from __future__ import annotations

from typing import Any


MAX_CONTENT_TOKENS = 254
WINDOW_OVERLAP_TOKENS = 32


def split_text_into_windows(
    text: str,
    tokenizer: Any,
    max_content_tokens: int = MAX_CONTENT_TOKENS,
    overlap_tokens: int = WINDOW_OVERLAP_TOKENS,
) -> list[dict]:
    """
    Split source text into overlapping embedding windows.

    Uses fast-tokenizer character offsets to preserve original text.
    Does not reconstruct source text by decoding token IDs.
    """

    if not text:
        return []

    if max_content_tokens <= 0:
        raise ValueError("max_content_tokens must be positive")

    if overlap_tokens < 0 or overlap_tokens >= max_content_tokens:
        raise ValueError(
            "overlap_tokens must be >= 0 and smaller than max_content_tokens"
        )

    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer with offset mappings is required")

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )

    offsets = encoded["offset_mapping"]
    total_tokens = len(encoded["input_ids"])

    if total_tokens == 0:
        return []

    windows = []
    start_token = 0
    window_index = 0

    while start_token < total_tokens:
        end_token = min(
            start_token + max_content_tokens,
            total_tokens,
        )

        start_char = (
            0
            if start_token == 0
            else offsets[start_token][0]
        )

        while True:
            end_char = (
                len(text)
                if end_token == total_tokens
                else offsets[end_token][0]
            )

            window_text = text[start_char:end_char]

            actual_token_count = len(
                tokenizer.encode(
                    window_text,
                    add_special_tokens=False,
                    truncation=False,
                )
            )

            if actual_token_count <= max_content_tokens:
                break

            end_token -= 1

            if end_token <= start_token:
                raise ValueError(
                    "Unable to create a window within the token budget"
                )

        if not window_text:
            raise ValueError(
                f"Empty window generated at index {window_index}"
            )

        windows.append(
            {
                "window_index": window_index,
                "start_token": start_token,
                "end_token": end_token,
                "start_char": start_char,
                "end_char": end_char,
                "token_count": actual_token_count,
                "character_count": len(window_text),
                "text": window_text,
            }
        )

        if end_token == total_tokens:
            break

        next_start = max(
            start_token + 1,
            end_token - overlap_tokens,
        )

        start_token = next_start
        window_index += 1

    return windows


def validate_windows(
    source_text: str,
    windows: list[dict],
    max_content_tokens: int = MAX_CONTENT_TOKENS,
) -> dict:
    """Validate exact source preservation and window token budgets."""

    if not source_text:
        return {
            "valid": not windows,
            "errors": [] if not windows else ["Unexpected windows for empty text"],
        }

    errors = []

    if not windows:
        return {
            "valid": False,
            "errors": ["No windows generated for non-empty source"],
        }

    if windows[0]["start_char"] != 0:
        errors.append("First window does not start at character zero")

    if windows[-1]["end_char"] != len(source_text):
        errors.append("Last window does not reach the end of source text")

    previous_end = 0

    for window in windows:
        start = window["start_char"]
        end = window["end_char"]

        if start < 0 or end > len(source_text) or start >= end:
            errors.append(
                f"Invalid character range in window {window['window_index']}"
            )
            continue

        if start > previous_end:
            errors.append(
                f"Source gap before window {window['window_index']}"
            )

        if window["text"] != source_text[start:end]:
            errors.append(
                f"Source text mismatch in window {window['window_index']}"
            )

        if window["token_count"] > max_content_tokens:
            errors.append(
                f"Token budget exceeded in window {window['window_index']}"
            )

        if window["character_count"] != len(window["text"]):
            errors.append(
                f"Character count mismatch in window {window['window_index']}"
            )

        previous_end = max(previous_end, end)

    if previous_end != len(source_text):
        errors.append("Not all source characters are covered")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
    }
