"""Finite synthetic drift experiment and optional concurrent inference load test."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import time

import numpy as np
import requests


def coefficients(seed):
    rng = np.random.RandomState(seed)
    return (rng.rand(10, 10), rng.rand(10, 10), rng.rand(10), rng.rand(10, 10))


def target(xs, params):
    a, b, c, d = params
    return ((a[None] * np.sin(b[None] * xs[:, None, :] ** 2)).sum(axis=2)
            + c[None] * np.cos(xs @ d.T))


def post(url, route, data):
    response = requests.post(url + route, json=data, timeout=30)
    response.raise_for_status()
    return response.json()


def wait_version(url, model_id, previous, timeout=300):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = requests.post(url + "/predict", json={"model_id": model_id, "input": [0] * 10}, timeout=30)
        if response.ok and response.json()["version"] > previous:
            return response.json()["version"]
        health = requests.get(url + "/health", timeout=5)
        if health.status_code == 503:
            raise RuntimeError("Trainer unavailable; inspect server logs")
        time.sleep(0.1)
    raise TimeoutError(f"Model {model_id} did not publish a new version")


def mse(url, model_id, xs, ys):
    result = post(url, "/predict", {"model_id": model_id, "inputs": xs.tolist()})
    return float(np.mean((np.asarray(result["prediction"]) - ys) ** 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:5000")
    parser.add_argument("--instances", type=int, default=1)
    parser.add_argument("--updates", type=int, default=30)
    parser.add_argument("--load-requests", type=int, default=0)
    parser.add_argument("--clients", type=int, default=16)
    parser.add_argument("--skip-training", action="store_true", help="Only load-test already trained model IDs")
    args = parser.parse_args()
    if min(args.instances, args.clients) < 1 or min(args.updates, args.load_requests) < 0:
        parser.error("instances/clients must be positive and updates/load-requests nonnegative")
    rng = np.random.RandomState(12345)
    report = []
    for model_id in range(0 if args.skip_training else args.instances):
        params = coefficients(model_id)
        xs = rng.uniform(-1, 1, (1000, 10))
        current = requests.post(args.url + "/predict", json={"model_id": model_id, "input": [0] * 10}, timeout=30)
        previous = current.json()["version"] if current.ok else 0
        post(args.url, "/train", {"model_id": model_id, "inputs": xs.tolist(),
                                   "outputs": target(xs, params).tolist()})
        version = wait_version(args.url, model_id, previous)
        holdout_x = rng.uniform(-1, 1, (512, 10))
        baseline = mse(args.url, model_id, holdout_x, target(holdout_x, params))
        # A fixed changed plant, evaluated on data never sent to the trainer.
        drifted = (params[0] + 0.15, params[1], params[2] + 0.1, params[3])
        holdout_y = target(holdout_x, drifted)
        frozen = mse(args.url, model_id, holdout_x, holdout_y)
        for _ in range(args.updates):
            new_x = rng.uniform(-1, 1, (64, 10))
            post(args.url, "/observe", {"model_id": model_id, "inputs": new_x.tolist(),
                                         "outputs": target(new_x, drifted).tolist()})
            version = wait_version(args.url, model_id, version)
        adapted = mse(args.url, model_id, holdout_x, holdout_y)
        row = {"model_id": model_id, "initial_holdout_mse": baseline,
               "frozen_drift_mse": frozen, "adapted_drift_mse": adapted,
               "improved": adapted < frozen, "version": version}
        report.append(row)
        print(json.dumps(row), flush=True)
    output = {"models": report}
    if args.load_requests:
        def predict(index):
            start = time.perf_counter()
            try:
                post(args.url, "/predict", {"model_id": index % args.instances, "input": [0.2] * 10})
                return time.perf_counter() - start, True
            except requests.RequestException:
                return time.perf_counter() - start, False
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.clients) as executor:
            results = list(executor.map(predict, range(args.load_requests)))
        elapsed = time.perf_counter() - start
        latencies = [duration * 1000 for duration, ok in results if ok]
        output["inference_load"] = {"requests": len(results), "clients": args.clients,
            "errors": sum(not ok for _, ok in results), "seconds": elapsed,
            "successful_requests_per_second": len(latencies) / elapsed,
            "p50_ms": float(np.percentile(latencies, 50)) if latencies else None,
            "p95_ms": float(np.percentile(latencies, 95)) if latencies else None}
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
