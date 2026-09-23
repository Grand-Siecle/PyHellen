# Benchmarks

Scripts used to measure the result cache and API responsiveness on a real corpus
(17th-century French OCR blocks, `freem` model). Corpus format: `{"doc": ["text block", ...]}`.

- `bench_cache.py` — `HybridCache` alone (no model): per-text cost and longest event-loop stall.
- `bench_api.py` — a running server over HTTP: throughput, request latency, and `/service/health`
  latency polled every 100 ms while tagging.

## Results (2026-09-17, RTX 4060 Laptop, WSL2, `main` @ 8eb562d vs this branch)

### Cache alone (`bench_cache.py`, LIV0001: 565 blocks, 24k tokens, median of repeats)

| Scenario | main | branch |
|---|---|---|
| New texts, one by one (`/tag`) | 5.5 ms/text, loop blocked 3.1 s | 5.2 ms/text, max stall 8 ms |
| Batches of 64 (`/batch`) | 5.5 ms/text, loop blocked 3.1 s | **0.49 ms/text**, max stall 12 ms |
| Hits read back from SQLite (restart) | 4.4 ms/text, loop blocked 2.5 s | **3.3 ms/text**, max stall 10 ms |
| Hits in memory | 0.006 ms/text | 0.006 ms/text |

### API end to end (`bench_api.py`, local uvicorn on GPU, 8 concurrent clients)

| Scenario | main | branch |
|---|---|---|
| `/tag` new texts: throughput | 212 tok/s | 213 tok/s (inference-bound) |
| `/tag` new texts: `/service/health` p50 / max | 915 ms / 2.9 s (61/141 probes > 1 s) | **6 ms / 146 ms** (0 > 1 s) |
| `/batch` new texts: `/service/health` p50 | 29 s (blocked per batch) | **5.5 ms** |
| `/tag` cached, short blocks (LIV0001) | 5.03 s, p50 68 ms | **3.88 s, p50 49 ms** |
| `/tag` cached, long blocks (150 LIV0002a blocks) | 1.76 s | **1.08 s** |
| `/tag` after restart (SQLite hits) | 6.52 s, 87 texts/s | **5.24 s, 108 texts/s** |

### Inference options (pie-extended 0.1.5 / PaPie 0.6.0, `freem`, 1 repetition)

GPU rows: LIV0001 then ~30k words of LIV0002a. CPU rows (container limited to 4 cores): first 250 LIV0001
blocks then ~8k words of LIV0002a. Cache runs never reached eviction (under 12k entries per sub-model).

| Device | Option | First document | Second document |
|---|---|---|---|
| CUDA | — | 272 tok/s | 475 tok/s |
| CUDA | PaPie char cache | 214 tok/s (−21%), VRAM ×3 | 409 tok/s (−14%) |
| CPU | — | 186 tok/s | 256 tok/s |
| CPU | PaPie char cache | **474 tok/s (×2.5)** | **729 tok/s (×2.8)** |
| CPU | INT8 quantization | 384 tok/s (×2, annotations change) | — |

Hence: PaPie's char cache is enabled on CPU only (`CHAR_CACHE_CPU_SIZE`), with a fix for its
eviction `KeyError` (validated on 100k tokens: no error, identical output). It is not used on GPU.

### Measured and rejected

Writing request logs from a worker thread instead of the event loop halved cached-request throughput
(184 → 103 texts/s; single writer thread: 130) because concurrent SQLite writers contend for the lock.
Without request logging the same workload reaches 294 texts/s, so batching log writes is the next
worthwhile step.

## Caveats

Single machine, WSL2; other GPU containers were idle but present. Inference numbers are single runs;
cache numbers are medians of 3–5 repeats. `main` cache runs overlapped a Docker image build.
