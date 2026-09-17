"""
End-to-end benchmark of a running PyHellen server over HTTP, on a realistic corpus.

For each scenario, every text block is tagged once and a background probe calls /service/health
every 100 ms, which measures how responsive the server stays while it tags.
  - /api/tag mode:   one request per block, N concurrent clients (teille-douce uses 8)
  - /api/batch mode: blocks sent in batches of B texts, N concurrent clients
Scenarios: "cold" (cache cleared first), "warm" (same texts again: cache hits). Run "warm" alone after
restarting the server to measure hits served from SQLite.

Usage:
    python benchmarks/bench_api.py --base-url http://localhost:8001 --corpus corpus.json \\
        --doc LIV0001_reconciled --mode tag --concurrency 8 --scenarios cold,warm --out results.json
"""

import argparse
import asyncio
import json
import time

import httpx


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q / 100 * (len(ordered) - 1))))
    return ordered[index]


def describe(values):
    return {
        "p50": round(percentile(values, 50), 1) if values else None,
        "p95": round(percentile(values, 95), 1) if values else None,
        "max": round(max(values), 1) if values else None,
    }


async def health_probe(client, stop, latencies, interval):
    while not stop.is_set():
        t0 = time.perf_counter()
        try:
            await client.get("/service/health", timeout=60)
            latencies.append((time.perf_counter() - t0) * 1000)
        except httpx.HTTPError:
            latencies.append(60_000.0)
        await asyncio.sleep(interval)


async def run_scenario(args, client, texts):
    if args.mode == "tag":
        jobs = [[text] for text in texts]
    else:
        jobs = [texts[i : i + args.batch_size] for i in range(0, len(texts), args.batch_size)]
    queue = list(enumerate(jobs))
    request_latencies, errors = [], []
    tokens = 0
    from_cache = 0

    async def worker():
        nonlocal tokens, from_cache
        while queue:
            _, job = queue.pop(0)
            t0 = time.perf_counter()
            if args.mode == "tag":
                response = await client.post(f"/api/tag/{args.model}", json={"text": job[0]}, timeout=600)
            else:
                response = await client.post(
                    f"/api/batch/{args.model}", json={"texts": job}, params={"concurrent": "false"}, timeout=600
                )
            request_latencies.append((time.perf_counter() - t0) * 1000)
            if response.status_code != 200:
                errors.append(response.status_code)
                continue
            data = response.json()
            if args.mode == "tag":
                tokens += len(data["result"])
                from_cache += bool(data.get("from_cache"))
            else:
                tokens += sum(len(result) for result in data["results"])
                from_cache += data.get("cache_hits", 0)

    stop = asyncio.Event()
    health_latencies = []
    probe = asyncio.create_task(health_probe(client, stop, health_latencies, args.health_interval))
    t0 = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(args.concurrency)))
    wall = time.perf_counter() - t0
    stop.set()
    await probe

    return {
        "wall_seconds": round(wall, 2),
        "texts_per_second": round(len(texts) / wall, 2),
        "tokens_per_second": round(tokens / wall, 1),
        "tokens": tokens,
        "texts_from_cache": from_cache,
        "errors": len(errors),
        "error_codes": sorted(set(errors)),
        "request_latency_ms": describe(request_latencies),
        "health_latency_ms": describe(health_latencies),
        "health_probes_over_1s": sum(latency > 1000 for latency in health_latencies),
        "health_probes": len(health_latencies),
    }


async def main_async(args):
    texts = json.load(open(args.corpus, encoding="utf-8"))[args.doc]
    if args.limit:
        texts = texts[: args.limit]

    report = {
        "label": args.label,
        "base_url": args.base_url,
        "doc": args.doc,
        "texts": len(texts),
        "mode": args.mode,
        "concurrency": args.concurrency,
        "batch_size": args.batch_size if args.mode == "batch" else None,
        "scenarios": {},
    }
    async with httpx.AsyncClient(base_url=args.base_url) as client:
        # Load the model outside the measurements
        warmup = await client.post(f"/api/tag/{args.model}", json={"text": "Warmup du modèle."}, timeout=1800)
        warmup.raise_for_status()

        for scenario in args.scenarios.split(","):
            if scenario == "cold":
                (await client.post("/api/cache/clear", timeout=120)).raise_for_status()
            report["scenarios"][scenario] = await run_scenario(args, client, texts)
            print(scenario, json.dumps(report["scenarios"][scenario]), flush=True)

        stats = await client.get("/api/cache/stats", timeout=60)
        report["cache_stats"] = stats.json() if stats.status_code == 200 else None

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--doc", default="LIV0001_reconciled")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model", default="freem")
    parser.add_argument("--mode", choices=["tag", "batch"], default="tag")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--health-interval", type=float, default=0.1)
    parser.add_argument("--scenarios", default="cold,warm")
    parser.add_argument("--label", default="")
    parser.add_argument("--out", default="")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
