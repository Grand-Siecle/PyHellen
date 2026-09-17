"""
Benchmark PyHellen's result cache (HybridCache) on a realistic workload.

Each text block of a corpus becomes one cached tagging result shaped like a pie-extended output.
Scenarios (fresh SQLite file per repeat, medians over repeats):
  - cold:        get (miss) + set for every block, sequentially (what /api/tag does on new texts)
  - warm_memory: get every block again from the same cache instance (in-memory hits)
  - warm_db:     get every block from a new cache instance on the same DB (restart: SQLite hits)
  - concurrent:  8 clients doing get+set on distinct blocks at once (teille-douce uses concurrency=8)
  - batch64:     cold pass in batches of 64 texts, as /api/batch does (get_many/set_many when available)
For each scenario we also run a 1 ms ticker coroutine and report the longest event-loop stall,
which shows whether cache I/O blocks other requests (health checks, concurrent clients).

Usage:
    python benchmarks/bench_cache.py --corpus corpus.json --doc LIV0001_reconciled --src . --out results.json
The corpus is a JSON object {doc_name: [text, ...]}. --src selects which PyHellen checkout to import.
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
import tempfile
import time


def make_result(text):
    return [
        {"form": w, "lemma": w.lower(), "POS": "NOMcom", "morph": "NOMB.=s|GENRE=m", "treated": w} for w in text.split()
    ]


class StallMeter:
    """Runs a 1 ms ticker and reports the longest gap between ticks (event-loop stall)."""

    def __init__(self):
        self.ticks = []
        self._task = None

    async def _run(self):
        while True:
            self.ticks.append(time.perf_counter())
            await asyncio.sleep(0.001)

    async def __aenter__(self):
        self._task = asyncio.create_task(self._run())
        await asyncio.sleep(0.005)
        return self

    async def __aexit__(self, *exc):
        await asyncio.sleep(0.005)
        self._task.cancel()

    @property
    def max_stall_ms(self):
        return max((b - a) * 1000 for a, b in zip(self.ticks, self.ticks[1:]))


def new_cache(HybridCache):
    return HybridCache(max_size=100_000, ttl_seconds=3600, persist=True)


async def run_repeat(HybridCache, reset_db, texts, results):
    out = {}
    reset_db()
    cache = new_cache(HybridCache)
    lower = False

    async with StallMeter() as meter:
        t0 = time.perf_counter()
        for text, result in zip(texts, results):
            if await cache.get("freem", text, lower) is None:
                await cache.set("freem", text, lower, result)
        out["cold_ms_per_text"] = (time.perf_counter() - t0) * 1000 / len(texts)
    out["cold_max_stall_ms"] = meter.max_stall_ms

    async with StallMeter() as meter:
        t0 = time.perf_counter()
        hits = sum([await cache.get("freem", text, lower) is not None for text in texts])
        out["warm_memory_ms_per_text"] = (time.perf_counter() - t0) * 1000 / len(texts)
    out["warm_memory_hit_rate"] = hits / len(texts)

    restarted = new_cache(HybridCache)
    async with StallMeter() as meter:
        t0 = time.perf_counter()
        hits = sum([await restarted.get("freem", text, lower) is not None for text in texts])
        out["warm_db_ms_per_text"] = (time.perf_counter() - t0) * 1000 / len(texts)
    out["warm_db_hit_rate"] = hits / len(texts)
    out["warm_db_max_stall_ms"] = meter.max_stall_ms

    reset_db()
    cache = new_cache(HybridCache)
    queue = list(zip(texts, results))

    async def client():
        while queue:
            text, result = queue.pop()
            if await cache.get("freem", text, lower) is None:
                await cache.set("freem", text, lower, result)

    async with StallMeter() as meter:
        t0 = time.perf_counter()
        await asyncio.gather(*(client() for _ in range(8)))
        out["concurrent8_ms_per_text"] = (time.perf_counter() - t0) * 1000 / len(texts)
    out["concurrent8_max_stall_ms"] = meter.max_stall_ms

    reset_db()
    cache = new_cache(HybridCache)
    bulk = hasattr(cache, "get_many")
    async with StallMeter() as meter:
        t0 = time.perf_counter()
        for start in range(0, len(texts), 64):
            chunk = list(zip(texts[start : start + 64], results[start : start + 64]))
            if bulk:
                found = await cache.get_many("freem", [text for text, _ in chunk], lower)
                await cache.set_many("freem", [item for item, hit in zip(chunk, found) if hit is None], lower)
            else:
                for text, result in chunk:
                    if await cache.get("freem", text, lower) is None:
                        await cache.set("freem", text, lower, result)
        out["batch64_ms_per_text"] = (time.perf_counter() - t0) * 1000 / len(texts)
    out["batch64_max_stall_ms"] = meter.max_stall_ms
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--doc", default="LIV0001_reconciled")
    parser.add_argument("--limit", type=int, default=0, help="Use only the first N blocks (0 = all)")
    parser.add_argument("--src", default=".", help="PyHellen checkout to benchmark")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--label", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    texts = json.load(open(args.corpus, encoding="utf-8"))[args.doc]
    if args.limit:
        texts = texts[: args.limit]
    results = [make_result(t) for t in texts]

    tmpdir = tempfile.mkdtemp(prefix="bench_cache_")
    db_path = os.path.join(tmpdir, "bench.db")
    os.environ["TOKEN_DB_PATH"] = db_path
    os.environ.setdefault("LOG_LEVEL", "ERROR")
    sys.path.insert(0, os.path.abspath(args.src))

    import logging

    logging.disable(logging.WARNING)
    from app.core.cache import HybridCache
    from app.core.database import get_db_engine
    from app.core.database.models import CacheEntry
    from sqlmodel import delete

    def reset_db():
        engine = get_db_engine()
        with engine.get_session() as session:
            session.exec(delete(CacheEntry))
            session.commit()

    runs = [asyncio.run(run_repeat(HybridCache, reset_db, texts, results)) for _ in range(args.repeats)]
    summary = {
        key: {
            "median": round(statistics.median(r[key] for r in runs), 4),
            "min": round(min(r[key] for r in runs), 4),
            "max": round(max(r[key] for r in runs), 4),
        }
        for key in runs[0]
    }
    report = {
        "label": args.label,
        "doc": args.doc,
        "texts": len(texts),
        "tokens": sum(len(r) for r in results),
        "repeats": args.repeats,
        "summary": summary,
    }
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
