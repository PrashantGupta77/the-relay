from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MAX_MODEL_SEQUENCE_LENGTH = 256
MAX_CONTENT_TOKENS = 254
WINDOW_OVERLAP_TOKENS = 32


def load_embedding_model():
    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )

    if model.max_seq_length != MAX_MODEL_SEQUENCE_LENGTH:
        raise RuntimeError(
            f"Unexpected model sequence length: {model.max_seq_length}"
        )

    return model


def load_offset_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    if not tokenizer.is_fast:
        raise RuntimeError("A fast tokenizer is required for source offsets.")

    # This tokenizer is used only to calculate offsets in the original text.
    # The SentenceTransformer model retains its own 256-token input limit.
    tokenizer.model_max_length = 1_000_000_000

    return tokenizer


def encode_texts(model, texts, batch_size=32):
    if not texts:
        raise ValueError("Cannot encode an empty list of texts.")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embeddings.astype("float32")
