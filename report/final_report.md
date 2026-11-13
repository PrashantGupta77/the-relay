# The Relay — ML Engineering Assignment

## 1. What I built and why

I built an evidence-grounded RAG service over three fixed paper versions: *Attention Is All You Need* (v7), *Moshi* (v1), and *Dynamic Chunking* (v2). It extracts PDF text, preserves metadata, creates overlapping token-based chunks, embeds them with `all-MiniLM-L6-v2`, and retrieves evidence using FAISS.

I compared chunk settings A (300/50), B (600/80), and C (900/120), representing chunk size/overlap. Answers must cite retrieved evidence; unsupported questions should be refused.

My key design decision was separating provider execution behind an abstraction. This let me test real Groq integration and a deterministic local stand-in without exhausting the real free-tier quota.

## 2. The real service and my stand-in

The real provider was Groq using `openai/gpt-oss-20b`. These are observed account quota values, not universal limits.

| Limit | Value found | How found | Confidence |
| --------------------- | -----------: | ----------------------------------- | ---------- |
| Requests per minute | 30 | Groq console snapshot | Medium |
| Tokens per minute | 8,000 | Groq console snapshot | Medium |
| Requests per day | 1,000 | Groq console snapshot | Medium |
| Tokens per day | 200,000 | Groq console snapshot | Medium |
| Requests in flight | Not measured | No confirmed concurrency limit | Low |
| Response when limited | Not measured | No deliberate real quota exhaustion | Low |
| Recovery time | Not measured | No deliberate real quota exhaustion | Low |

The stand-in exposes configurable latency, TTFT, concurrency, overload slowdown, failures, RPM, TPM, and daily budgets. Its defaults include 500 ms base latency, 250 ms TTFT, concurrency 8, and simulated usage of 1,200 input/180 output tokens. The client allows two retries within a 30-second deadline.

It is a deterministic simulation, not a language model. Its token counts and answers are not real generation measurements. Its daily counter is in-memory and does not reset automatically at midnight.

## 3. Results

All stand-in percentiles use nearest-rank calculation. Quota-scenario latency percentiles include successful requests only.

### Table 2a — Stand-in latency and load

| Scenario | Requests | Duration (s) | Offered rate | Achieved req/s | TTFT P50/P95/P99 (ms) | Total P50/P95/P99 (ms) |
| ---------------------- | -------: | -----------: | ------------- | -------------: | --------------------- | ---------------------- |
| Steady | 12 | 6.71 | 120/min | 1.79 | 754/780/780 | 1,103/1,122/1,122 |
| Burst | 24 | 12.28 | Unpaced burst | 1.96 | 5,792/11,425/11,949 | 6,165/11,808/12,263 |
| Sustained overload | 30 | 16.55 | 300/min | 1.81 | 5,258/9,846/10,307 | 5,633/10,219/10,696 |
| RPM halved | 18 | 4.11 | Unpaced burst | 4.38 | 803/803/803 | 1,230/1,284/1,284 |
| Daily budget exhausted | 12 | 1.26 | Unpaced burst | 9.52 | 855/855/855 | 1,229/1,255/1,255 |

### Table 2b — Stand-in outcomes

| Scenario | Succeeded | Rate-limited | Timed out/errors | Avg retries/request | Peak queue | Tokens in/out | Cost per 1,000 answers |
| ---------------------- | --------: | -----------: | ---------------: | ------------------: | ---------: | ------------- | ---------------------- |
| Steady | 100.0% | 0.0% | 0.0% | 0.00 | 1 | 1,200/180* | Not measured |
| Burst | 100.0% | 0.0% | 0.0% | 0.00 | 24 | 1,200/180* | Not measured |
| Sustained overload | 100.0% | 0.0% | 0.0% | 0.00 | 12 | 1,200/180* | Not measured |
| RPM halved | 83.3% | 16.7% | 0.0% | 0.33 | 18 | 1,200/180* | Not measured |
| Daily budget exhausted | 66.7% | 33.3% | 0.0% | 0.00 | 12 | 1,200/180* | Not measured |

Across 96 stand-in requests, 89 succeeded and seven were rate-limited; six retries occurred. No request timed out in these runs. This reflects the configured client deadline, not proof that real users would wait indefinitely. Queue values are client-side load-runner measurements. *Token counts are simulated.*

### Real-provider observations

Three successful streaming-calibration requests had median TTFT 765 ms and median total latency 795 ms. Three samples are insufficient for reliable tail-percentile claims.

Setting B generation evaluation recorded 21 real Groq attempts across all 20 benchmark questions (19 answerable and one unanswerable). ATT-03 was attempted twice. End-to-end TTFT P50/P95 was 1,984/14,846 ms across 20 samples; one attempt produced no first visible token. End-to-end total latency P50/P95 was 4,794/18,483 ms across all 21 attempts. Mean usage was 2,742 input and 402 output tokens per attempt, or 66,011 tokens in total.

After grouping by unique question, 12 of 19 answerable questions produced correct, cited answers (63.2%). Two were factually wrong (10.5%). Three answers were factually plausible but rejected because citation validation failed (15.8%). Two generations were incomplete due to output truncation (10.5%). The one unanswerable question was correctly refused.

ATT-03 had a truncated attempt followed by a separately initiated successful attempt. DYN-01 and MOS-04 were factually wrong. ATT-02, MOS-03, and DYN-05 contained factually plausible answers but failed citation validation. In ATT-02, the answer's explanation was supported by the retrieved passage in Section 3.2.2, but the generated citation references did not resolve to evidence: the citation IDs and citation records were empty. MOS-05 and CROSS-01 were incomplete generations. These results distinguish factual correctness, citation validity, and pipeline acceptance; a successful pipeline status alone does not establish answer correctness.

## 4. Where the time goes

The stand-in shows queueing and overload slowdown can dominate latency. Burst TTFT P95 reached 11.4 seconds; sustained-overload TTFT P95 reached 9.8 seconds.

One slow burst request, `f1e83499-d60d-4fdb-9842-c5e898288778`, had TTFT 5,280.76 ms and total latency 5,641.01 ms, with no retry. The 360.25 ms after first content is the remaining response time. This trace does not separately measure queue wait and model-working time, so I cannot reliably split the TTFT between them.

Real RAG latency also includes retrieval, prompt construction, provider generation, and streaming. My small sample does not isolate each stage.

## 5. When a limit is hit

The stand-in returns HTTP 429 for RPM, TPM, and daily-budget exhaustion. RPM/TPM responses include `Retry-After`; daily-budget exhaustion does not. Eligible transient failures and rate limits can be retried up to two times within the deadline. Daily-budget exhaustion is not retried.

In the RPM test, three logical requests remained rate-limited after retries because the rolling window was still full. This illustrates why blindly retrying a rate limit can add latency without restoring capacity.

### Retrieval: quality versus cost

The retrieval benchmark contains 20 questions: 19 answerable and one unanswerable. Each setting used top-five retrieval without a reranker.

| Setting (chunk size/overlap; fetched; rerank) | Questions asked | Correct and cited | Wrong | Citation validation failures | Incomplete/truncated | Correctly refused | Avg tokens per attempt (in/out) | TTFT P50/P95 (ms) | Total P50/P95 (ms) |
| --------------------------------------------- | ----------------------------------: | ----------------: | -----------: | ---------------------------: | -------------------: | ----------------: | ------------------------------- | ----------------- | ------------------ |
| A (300/50; 5; no) | 20 retrieval; 0 generated | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured |
| B (600/80; 5; no) | 20 retrieval; 21 attempts/20 unique | 63.2% (12/19) | 10.5% (2/19) | 15.8% (3/19) | 10.5% (2/19) | 100.0% (1/1) | 2,742/402 | 1,984/14,846 | 4,794/18,483 |
| C (900/120; 5; no) | 20 retrieval; 0 generated | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured |

Retrieval mean relevance was A 0.884, B 0.932, and C 0.895. Direct evidence hits were 13/19 for A and 16/19 for both B and C; all three returned at least one relevant passage for 19/19 answerable questions. These relevance grades are not answer-accuracy scores. Generation was evaluated only with Setting B; A/C generation metrics were not measured.

When approaching a quota, a possible mitigation is reducing top-k to shorten the prompt, while accepting a risk of losing supporting evidence. I did not load-test this mitigation.

### How this library will mislead a naive retriever

- **Moshi Helium question:** Asking which section explains Moshi's Helium foundation can mislead section-number matching. The expected evidence is in Section 3.1, page 7; the corresponding Section 3.2 chunk contained only its heading.
- **Attention ATT-02:** The expected evidence is in Section 3.2.2. The initial retrieval ranked the neighbouring Section 3.2.3 passage higher, despite the expected evidence being in 3.2.2.
- **Cross-paper CROSS-01:** A question requiring evidence across Attention and Dynamic Chunking retrieves top-five results separately from both papers. The resulting ten candidates can crowd out the most useful supporting passage.

PDF reading order, tables, long sections, and chunk boundaries also affect retrieval. I did not implement OCR or table reconstruction.

## 6. Retries

Retries are recorded as individual attempts in `traces.jsonl`; `results.csv` stores one row per logical request. The Groq SDK's automatic retries were disabled during calibration.

The RPM request `6e2dce06-f2bf-417b-8110-f75d28b6a6d1` ended rate-limited after two retries, with total latency 4,070.04 ms. The retries increased waiting but did not recover the request. This is why retries need limits, deadlines, and observability rather than being treated as a guarantee of success.

## 7. What the person sees

When sufficient evidence exists, the service should stream an answer with evidence references. When evidence is insufficient, it should clearly refuse rather than guess.

At 3 pm, if the daily model budget is exhausted, a waiting person receives: **“Daily model quota exhausted. A verified answer is temporarily unavailable.”** The service does not retry this daily-budget rejection or invent an answer. This is proposed user-facing wording, not a captured UI response.

A successful pipeline status is not a factuality guarantee: manual review found accepted answers that were factually incorrect. Invalid citation formats can also cause an otherwise plausible answer to be rejected.

## 8. What I'm unsure about, what I left out, what I learned

My least-certain stand-in assumption is fixed usage of 1,200 input/180 output tokens. Larger real prompts could exhaust TPM sooner and change queueing behaviour. I did not establish Groq concurrency, real recovery time, or cost per 1,000 answers. I also omitted OCR/table reconstruction and a full live-provider load matrix because of free-tier quota constraints.

The evaluation is small and manually graded. Pytest discovered no automated test cases; I do not claim a passing automated test suite. Validation was performed through scripts, recorded checks, and manual review.

The main lesson is that retrieval relevance, citation validity, pipeline status, and factual correctness are different measurements. A system must not treat one as proof of another.

## Your feedback on Vaani

The assignment connected RAG quality with serving concerns such as quotas, queueing, streaming, retries, and observability. I found the practical, measurement-led approach valuable. Clearer documentation of the boundary between the self-serve development workflow and production/managed deployment would make the offering easier to evaluate.

## How I worked with the assistant

I used ChatGPT for stepwise PowerShell guidance, debugging, experiment design, and report review. Key prompts asked it to guide implementation with exact Windows commands, design the load scenarios, and audit CSV/trace evidence before reporting metrics.

It caught an unsupported Groq `stream_options` parameter that caused a client-side error. I overruled the suggestion to deliberately exhaust the live free-tier quota and used a local stand-in instead. I wrote and ran the implementation, extracted and chunked the papers, built indexes, ran experiments, inspected traces, and manually graded answers. The measurements come from project artefacts, not assistant-generated estimates. The conversation history is provided separately.