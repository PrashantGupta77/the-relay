# The Relay — Evidence-Grounded RAG and LLM Reliability Evaluation

The Relay is an experimental Retrieval-Augmented Generation (RAG) and LLM reliability evaluation system built for the Vaani ML Engineering assignment.

It combines a document retrieval pipeline over three fixed research-paper versions with a hosted Groq LLM provider and a configurable local stand-in provider.

The project investigates two related areas:

- **Retrieval and answer evaluation:** document processing, chunking, embedding generation, FAISS retrieval, evidence grounding, and generated-answer review.
- **Provider reliability and load evaluation:** latency, time to first token (TTFT), request outcomes, rate limits, concurrency, queue behavior, and simulated quota constraints.

An important distinction throughout the project is the separation of **real hosted-provider observations** from **controlled local stand-in simulations**. Simulated results are not presented as measurements of Groq or production-provider behavior.

---

## 1. Project objectives

### Evidence-grounded question answering

- Process fixed versions of research papers.
- Preserve document, page, and section metadata.
- Compare multiple overlapping chunk configurations.
- Generate local embeddings and build FAISS indexes.
- Retrieve relevant passages for a question.
- Generate answers using a hosted LLM and retain supporting evidence.

### LLM provider reliability and load evaluation

- Measure TTFT and total request latency.
- Record request outcomes, errors, and rate-limit responses.
- Exercise steady, burst, and sustained-overload traffic.
- Simulate reduced requests-per-minute capacity and exhausted daily request budgets.
- Record request-level traces and relevant timing information.
- Compare real-provider observations with controlled stand-in scenarios.

## 2. Research-paper corpus

The retrieval corpus uses these fixed document versions:

| Document | Version |
|---|---|
| Attention Is All You Need | v7 |
| Moshi | v1 |
| Dynamic Chunking | v2 |

The document-processing pipeline preserves source information, including document identity, page information, and section metadata. This supports tracing retrieved passages back to their source.

The project uses local embeddings rather than requiring a hosted embedding API.

## 3. Architecture

```text
                  RESEARCH PAPER PDFs
                           |
                           v
               PDF EXTRACTION AND CLEANING
                           |
                           v
                SECTION AND METADATA PARSING
                           |
                           v
                 TOKEN-BASED CHUNKING
                     A / B / C
                           |
                           v
                 LOCAL EMBEDDING MODEL
                           |
                           v
                     FAISS INDEX
                           |
                           v
USER QUESTION ---> QUERY EMBEDDING AND RETRIEVAL
                           |
                           v
                   RETRIEVED EVIDENCE
                  + SOURCE METADATA
                           |
                           v
                    CONTEXT ASSEMBLY
                           |
                           v
                    PROVIDER INTERFACE
                           |
                 +---------+---------+
                 |                   |
                 v                   v
            GROQ PROVIDER       LOCAL STAND-IN
                 |                   |
                 v                   v
          HOSTED LLM API      CONFIGURABLE SIMULATION
                 |                   |
                 +---------+---------+
                           |
                           v
                  RESPONSE / OUTCOME
                           |
                           v
                 REQUEST LOGGING
                    AND TRACING
                           |
                 +---------+---------+
                 |                   |
                 v                   v
             RESULTS CSV        JSONL TRACES
                 |
                 v
          MANUAL ANSWER REVIEW
```

### Main application components

| Component | Responsibility |
|---|---|
| `app/retrieval/` | PDF processing, text cleaning, section parsing, chunking, embeddings, indexing, and retrieval. |
| `app/pipeline/` | Orchestrates retrieval, context construction, and answer generation. |
| `app/providers/` | Common provider interface, Groq integration, and stand-in client. |
| `app/standin/` | Local HTTP service for controlled provider simulations. |
| `app/evaluation/` | Evaluation questions, request logging, and load-test logging. |
| `experiments/` | Experiment runners, comparisons, diagnostics, test-named scripts, and validation scripts. |
| `data/metadata/` | Paper and index metadata manifests. |
| `results/` | Tracked experiment outputs, request records, traces, and evaluation artifacts. |
| `report/` | Experiment notes and final assignment report. |

## 4. Repository structure

The following structure reflects the tracked project files. Generated chunks and FAISS index files may be present locally when the pipeline has been run, but they are not listed as tracked repository files.

```text
The_Relay/
│
├── app/
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── load_logger.py
│   │   ├── questions.py
│   │   └── request_logger.py
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── orchestrator.py
│   │
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── groq_provider.py
│   │   └── standin_client.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── chunker.py
│   │   ├── embedding_windows.py
│   │   ├── embeddings.py
│   │   ├── index.py
│   │   ├── pdf_parser.py
│   │   ├── retriever.py
│   │   ├── section_parser.py
│   │   └── text_cleaner.py
│   │
│   └── standin/
│       ├── __init__.py
│       └── service.py
│
├── data/
│   └── metadata/
│       ├── chunking_summary.json
│       ├── paper_manifest.json
│       ├── paper_manifest_sorted.json
│       ├── setting_a_index_manifest.json
│       ├── setting_b_index_manifest.json
│       └── setting_c_index_manifest.json
│
├── experiments/
│   ├── analyse_query_expansion.py
│   ├── build_context_index.py
│   ├── build_faiss_indexes.py
│   ├── check_groq_models.py
│   ├── check_section_metadata.py
│   ├── compare_chunk_granularity.py
│   ├── compare_context_ranking.py
│   ├── compare_pdf_extraction.py
│   ├── compare_query_expansion.py
│   ├── create_manual_grading_template.py
│   ├── diagnose_dynamic_ranking.py
│   ├── first_real_request.py
│   ├── inspect_*.py
│   ├── run_load_test.py
│   ├── run_rag_answer_evaluation.py
│   ├── run_retrieval_benchmark.py
│   ├── sequential_calibration.py
│   ├── streaming_calibration.py
│   ├── summarise_retrieval_baseline.py
│   ├── test_embedding_generation.py
│   ├── test_embedding_windows.py
│   ├── test_rag_groq.py
│   ├── test_retrieval.py
│   ├── test_section_reranking.py
│   ├── test_standin_client.py
│   ├── validate_chunks.py
│   ├── validate_faiss_indexes.py
│   ├── validate_question_sources.py
│   ├── validate_source_coverage.py
│   ├── verify_cleaning.py
│   └── verify_section_parser.py
│
├── report/
│   ├── experiment_notes.md
│   └── final_report.md
│
├── results/
│   ├── chunk_granularity_comparison.json
│   ├── context_ranking_comparison.json
│   ├── first_real_request.json
│   ├── model_info.json
│   ├── query_expansion_analysis.csv
│   ├── query_expansion_setting_b.jsonl
│   ├── rag_answer_evaluation.jsonl
│   ├── results.csv
│   ├── retrieval_baseline_live.txt
│   ├── retrieval_benchmark_raw.jsonl
│   ├── retrieval_evidence_review_setting_b.txt
│   ├── retrieval_full_baseline.txt
│   ├── retrieval_manual_grading.csv
│   ├── retrieval_smoke_test.json
│   ├── section_reranking_experiment.json
│   └── traces.jsonl
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

The `inspect_*.py` scripts represent the tracked diagnostic scripts in `experiments/`. The repository also contains additional tracked experiment and validation scripts beyond the representative names shown above.

Local `.env` files, virtual environments, timestamped backups, and untracked temporary files are not part of this repository structure.

## 5. Retrieval pipeline

### 5.1 Document extraction and preparation

The source PDFs are processed into page-level text records. The preparation workflow includes:

1. PDF text extraction.
2. Preservation of original extracted text.
3. Text cleaning for retrieval.
4. Section and heading parsing.
5. Preservation of document, page, and section metadata.
6. Token-based chunk generation with overlap.

The pipeline retains source metadata to support evidence traceability.

### 5.2 Chunk configurations

Three chunk configurations were compared:

| Setting | Chunk size | Overlap |
|---|---:|---:|
| A | 300 tokens | 50 tokens |
| B | 600 tokens | 80 tokens |
| C | 900 tokens | 120 tokens |

These configurations represent different trade-offs:

- **Setting A:** smaller passages provide more granular retrieval.
- **Setting B:** medium-sized passages balance local context and retrieval granularity.
- **Setting C:** larger passages retain more surrounding context but may return less focused evidence.

The generated chunk data is used by the retrieval and indexing workflow.

### 5.3 Embeddings and vector indexes

The embedding model is:

`sentence-transformers/all-MiniLM-L6-v2`

The generated embeddings are indexed locally using FAISS.

Index files and their associated chunk data are generated by the project scripts. The tracked metadata manifests for Settings A, B, and C are stored under `data/metadata/`.

The retrieval workflow loads the selected setting's index and associated chunk metadata, embeds the query, and retrieves relevant evidence passages.

### 5.4 Retrieval benchmarking

The retrieval benchmark compares Settings A, B, and C against the evaluation question set.

The evaluation examines:

- Whether relevant evidence appears in retrieved results.
- The rank of expected evidence.
- Retrieval relevance scores.
- Whether retrieved passages contain sufficient context to answer a question.

The benchmark and its output artifacts are separate from generated-answer evaluation.

To inspect the available benchmark options:

```powershell
python -m experiments.run_retrieval_benchmark --help
```

### 5.5 Query expansion

The project includes query-expansion experiments to examine whether reformulating a question improves the rank of relevant evidence.

These experiments are separate from the baseline retrieval comparison. Improvements in retrieval rank should not automatically be interpreted as improvements in generated-answer correctness or factual accuracy.

## 6. Provider architecture

The project separates provider-specific behavior behind a common provider interface.

### 6.1 Groq provider

The hosted provider uses Groq with the configured model:

`openai/gpt-oss-20b`

The Groq provider is used for real LLM request and streaming calibration, as well as RAG answer-generation evaluation.

The account limits observed during the assignment experiments were:

| Limit | Observed value |
|---|---:|
| Requests per minute | 30 |
| Tokens per minute | 8,000 |
| Requests per day | 1,000 |
| Tokens per day | 200,000 |

These values describe the limits observed during the experiments. They should not be treated as universal or permanent Groq limits.

Real-provider measurements are subject to network conditions, model availability, provider-side rate limits, and account quotas.

### 6.2 Local stand-in provider

The local stand-in provides a controlled environment for reliability and load experiments without consuming real provider quota.

Its configurable behavior includes:

- Base request latency.
- Time-to-first-token behavior.
- Concurrent request capacity.
- Simulated failures and rate limits.
- Requests-per-minute and tokens-per-minute limits.
- Daily request-budget exhaustion.
- Streaming-style response behavior.
- Queue-wait observations.

The stand-in supports repeatable experiments under controlled conditions. It is not a replica of Groq's internal scheduling, infrastructure, or rate-limit implementation.

### 6.3 Why both providers are used

| Groq provider | Local stand-in |
|---|---|
| Real external API requests | Controlled local simulation |
| Observed provider latency | Configurable latency |
| Actual account quotas | Configurable request and token limits |
| Network and provider variability | Repeatable test conditions |
| Real responses and token usage | Simulated workload behavior |

Keeping the two sources separate avoids treating simulated behavior as real-provider evidence.

## 7. Reliability and load-testing workflow

The load-testing workflow exercises the stand-in under multiple traffic and quota conditions.

The scenarios include:

| Scenario | Purpose |
|---|---|
| `steady` | Observe behavior under regular request traffic. |
| `burst` | Examine a short increase in concurrent demand. |
| `sustained_overload` | Examine behavior under continued high demand. |
| `rpm_halved` | Simulate a reduced requests-per-minute allowance. |
| `daily_budget` | Simulate exhaustion of the available daily request budget. |

The experiments record request outcomes and latency information, including:

- Request count and successful requests.
- Rate-limited requests and errors.
- Retry count.
- TTFT and total latency.
- Throughput.
- Queue-related observations.

Inspect the load-test runner's available options with:

```powershell
python -m experiments.run_load_test --help
```

The exact scenario options and configuration flags are exposed by the script's help output.

**Measurement distinction:** stand-in scenarios are controlled simulations. Their results demonstrate how the application behaves under configured conditions; they do not establish how the real Groq service behaves under an identical artificial workload.

## 8. RAG answer evaluation

The answer-evaluation workflow uses a question set covering the source papers and cross-document questions.

The evaluation set contains 20 questions:

- 19 answerable questions.
- 1 intentionally unanswerable question.

The project includes retrieval comparisons across Settings A, B, and C. The real Groq answer-generation evaluation was conducted using **Setting B**. Retrieval-only benchmark results for Settings A and C should not be described as equivalent real-provider answer-generation evaluations.

An example invocation is:

```powershell
python -m experiments.run_rag_answer_evaluation --question ATT-05 --setting b
```

The script's help output should be consulted for the currently supported arguments and options.

Individual real-provider requests are kept traceable. Failed requests should not be silently treated as successful evaluations, and retry behavior should be interpreted according to the actual run configuration.

### Manual grading

Generated answers are reviewed against the retrieved source evidence.

Manual review distinguishes outcomes such as:

- Correct and adequately supported.
- Factually incorrect.
- Citation-validation failure.
- Incomplete or truncated answer.
- Correct refusal for an unanswerable question.

Where the assignment's CSV schema requires a restricted grade vocabulary, the exported grade must follow that schema. More detailed review reasons should be retained in the accompanying evaluation records or report rather than introducing unsupported CSV enum values.

A correct refusal is not the same as a provider failure. Similarly, a citation-validation failure does not automatically mean the underlying factual statement is false.

## 9. Request logging and result artifacts

The project records request-level information to support traceability and analysis.

Important tracked output files include:

| File | Purpose |
|---|---|
| `results/results.csv` | Consolidated experiment and request-level results. |
| `results/traces.jsonl` | Request traces and timing information. |
| `results/rag_answer_evaluation.jsonl` | RAG answer-evaluation records. |

Other tracked results include retrieval benchmarks, query-expansion analysis, context-ranking comparisons, chunk-granularity comparisons, calibration outputs, and manual retrieval-review artifacts.

Individual scripts may append records to result files. Check the behavior of a script before rerunning it, so existing evaluation records are not unintentionally duplicated or overwritten.

Before submitting results, verify:

- Expected columns and fields are present.
- Request IDs are unique where required.
- Real-provider and stand-in records are distinguishable.
- Smoke tests and diagnostic requests are not incorrectly counted as evaluation records.
- Reported metrics can be recomputed from the submitted records.
- Grading values comply with the assignment's required schema.

## 10. Setup

### Prerequisites

- Python 3.11 recommended.
- Git.
- Internet connectivity for package installation and real Groq requests.
- A Groq API key for hosted-provider experiments.

### Clone the repository

```powershell
git clone https://github.com/PrashantGupta77/the-relay.git
cd the-relay
```

### Create a virtual environment

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks environment activation, use an appropriate local execution-policy setting or invoke the virtual environment's Python executable directly.

### Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Configure environment variables

Create a local `.env` file from the example:

```powershell
Copy-Item .env.example .env
```

Add the required provider credentials and configuration values, following the variable names and placeholders documented in `.env.example` and the application configuration.

**Security precautions:**

- Never commit `.env`.
- Never put API keys directly into source files.
- Do not include credentials in exported logs, screenshots, or submission artifacts.
- Keep `.env.example` limited to variable names and safe placeholder values.

## 11. Running the project

### A. Inspect experiment options

```powershell
python -m experiments.run_retrieval_benchmark --help
python -m experiments.run_load_test --help
python -m experiments.run_rag_answer_evaluation --help
```

### B. Build FAISS indexes

```powershell
python -m experiments.build_faiss_indexes
```

Run index building when the underlying chunks or embedding/index configuration need to be rebuilt. Avoid rebuilding validated indexes unnecessarily during final evaluation.

### C. Run the local stand-in service

```powershell
uvicorn app.standin.service:app --host 127.0.0.1 --port 8000
```

The stand-in service runs locally and exposes its API for controlled simulations. Its interactive API documentation can be accessed at:

`http://127.0.0.1:8000/docs`

### D. Run stand-in load experiments

Inspect the available arguments:

```powershell
python -m experiments.run_load_test --help
```

Run the required scenario using the options exposed by the script.

### E. Run a real Groq RAG evaluation

Example:

```powershell
python -m experiments.run_rag_answer_evaluation --question ATT-05 --setting b
```

Replace the question ID with the intended evaluation input. The real answer-generation results documented for this assignment use Setting B.

### F. Inspect generated results

Review the main output files:

```text
results/results.csv
results/traces.jsonl
results/rag_answer_evaluation.jsonl
```

Check whether an experiment appends records before rerunning it.

## 12. Testing and validation

There is no dedicated `tests/` directory in the tracked repository.

Instead, the `experiments/` directory contains several test-named scripts, including:

- `experiments/test_embedding_generation.py`
- `experiments/test_embedding_windows.py`
- `experiments/test_rag_groq.py`
- `experiments/test_retrieval.py`
- `experiments/test_section_reranking.py`
- `experiments/test_standin_client.py`

It also contains separate validation scripts, such as:

- `experiments/validate_chunks.py`
- `experiments/validate_faiss_indexes.py`
- `experiments/validate_question_sources.py`
- `experiments/validate_source_coverage.py`
- `experiments/verify_cleaning.py`
- `experiments/verify_section_parser.py`

These scripts serve different purposes: some exercise specific components, while others validate generated data or inspect experiment behavior. They should not all be treated as a single automated test suite.

For example, the stand-in client script can be invoked with:

```powershell
python -m experiments.test_standin_client
```

Run only the checks appropriate to the current environment. Report a test or validation as successful only when it has actually been executed and passed.

## 13. Technology stack

| Area | Technology |
|---|---|
| Language | Python |
| API / local service | FastAPI |
| Hosted LLM provider | Groq |
| LLM model | `openai/gpt-oss-20b` |
| Embeddings | Sentence Transformers |
| Embedding model | `all-MiniLM-L6-v2` |
| Vector search | FAISS |
| Data processing | Python, JSON, JSONL, CSV |
| HTTP client | HTTPX |
| Experimentation | Python scripts |
| Version control | Git |

## 14. Reproducibility and limitations

For a reproducible evaluation:

1. Use the same source-paper versions.
2. Use the same chunk settings and corresponding indexes.
3. Keep the embedding model consistent.
4. Record the retrieval setting for each answer-generation run.
5. Separate real-provider and stand-in observations.
6. Preserve request IDs and timestamps.
7. Record retries and rate-limit outcomes explicitly.
8. Keep manual grading tied to the generated answer and its evidence.
9. Recompute report metrics from the final CSV and trace files.
10. Document unmeasured behavior rather than estimating it.

### Known limitations

- Real-provider measurements are affected by external network and provider conditions.
- The local stand-in models configured behavior, not the internal implementation of a hosted provider.
- Retrieval relevance does not guarantee a factually correct generated answer.
- A retrieved passage may be relevant but insufficient to support every statement in an answer.
- Citation validation and factual correctness are related but distinct evaluation dimensions.
- Retrieval-only benchmark results do not establish answer-generation performance for the same setting.
- The real Groq answer-generation evaluation documented for this assignment used Setting B; equivalent answer-generation coverage should not be inferred for Settings A and C.
- A limited number of real-provider requests cannot establish production-scale reliability.
- No production deployment or production-traffic claim is made by this assignment implementation.

## 15. Engineering decisions

The project emphasizes measurement and traceability over unsupported performance claims.

Key engineering decisions include:

- Using a common provider interface to separate application logic from provider-specific behavior.
- Keeping a local stand-in for repeatable quota and overload simulations.
- Recording request-level traces instead of relying only on aggregate averages.
- Comparing multiple chunk sizes rather than assuming one configuration is optimal.
- Preserving source metadata so retrieved evidence can be inspected.
- Separating retrieval benchmarking from answer-generation evaluation.
- Keeping real-provider observations distinct from simulation results.
- Treating incomplete outputs and citation-validation failures as explicit review outcomes.

The central engineering lesson is that an LLM application must be evaluated at multiple layers: retrieval quality, evidence sufficiency, answer correctness, citation validity, provider latency, and quota behavior all affect the reliability of the final system.

## 16. Assignment information

**Project:** The Relay  
**Purpose:** Vaani ML Engineering Intern Assignment  
**Repository:** [PrashantGupta77/the-relay](https://github.com/PrashantGupta77/the-relay)