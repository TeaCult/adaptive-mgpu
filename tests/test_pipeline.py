import multiprocessing as mp
from queue import Queue
import time

import numpy as np
import pytest
import torch

from adaptive_mgpu.demo import coefficients, target
from adaptive_mgpu.model import MIMOModel
from adaptive_mgpu.server import Config, create_app, validate_devices
from adaptive_mgpu.trainer import checkpoint_path, train_worker


@pytest.fixture
def pipeline(tmp_path):
    config = Config(infer_device="cpu", train_device="cpu", instances=200,
                    checkpoints=str(tmp_path), initial_steps=80, update_steps=20)
    context = mp.get_context("spawn")
    jobs, stop = context.Queue(16), context.Event()
    worker = context.Process(target=train_worker, args=(config, jobs, stop))
    worker.start()
    app = create_app(config, jobs, worker.is_alive)
    yield config, app.test_client(), worker
    stop.set()
    worker.join(10)
    if worker.is_alive():
        worker.terminate()
        worker.join(5)
    jobs.close()
    jobs.cancel_join_thread()


def wait(client, model_id=0, previous=0):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        response = client.post("/predict", json={"model_id": model_id, "input": [0.2] * 10})
        if response.status_code == 200 and response.json["version"] > previous:
            return response.json
        time.sleep(0.05)
    pytest.fail("Training worker failed to publish")


def test_independent_models_live_publication_and_restore(pipeline):
    config, client, worker = pipeline
    rng = np.random.RandomState(42)
    xs = rng.uniform(-1, 1, (128, 10))
    assert client.post("/predict", json={"input": [0] * 10}).status_code == 503
    for model_id, value in ((0, 0.0), (199, 2.0)):
        response = client.post("/train", json={"model_id": model_id, "inputs": xs.tolist(),
                                                "outputs": np.full((128, 10), value).tolist()})
        assert response.status_code == 202
    zero, two = wait(client), wait(client, 199)
    assert np.mean(np.square(zero["prediction"])) < 0.01
    # This integration test checks independent learning, not convergence to a
    # specific unseen point after a small, fixed number of optimizer steps.
    assert np.mean(two["prediction"]) > np.mean(zero["prediction"]) + 0.5
    assert client.post("/observe", json={"inputs": xs.tolist(), "outputs": np.ones((128, 10)).tolist()}).status_code == 202
    # The other model continues to serve while adaptation runs.
    unchanged = client.post("/predict", json={"model_id": 199, "input": [0.2] * 10}).json
    assert unchanged == two
    changed = wait(client, previous=zero["version"])
    assert changed["version"] == zero["version"] + 1
    assert np.mean(changed["prediction"]) > np.mean(zero["prediction"]) + 0.1
    # A new inference app loads the exact published state.
    restored = create_app(config, Queue(1), lambda: True).test_client()
    assert restored.post("/predict", json={"input": [0.2] * 10}).json == changed
    # Resume in a newly spawned trainer, including Adam state and version.
    worker.terminate()
    worker.join(5)
    context = mp.get_context("spawn")
    jobs, stop = context.Queue(1), context.Event()
    resumed = context.Process(target=train_worker, args=(config, jobs, stop))
    resumed.start()
    try:
        jobs.put((0, xs.tolist(), np.ones((128, 10)).tolist(), False))
        assert wait(restored, previous=changed["version"])["version"] == changed["version"] + 1
    finally:
        stop.set()
        resumed.join(10)
        if resumed.is_alive():
            resumed.terminate()
            resumed.join(5)
        jobs.close()
        jobs.cancel_join_thread()


@pytest.mark.parametrize("data", [None, [], {}, {"input": [0] * 9},
    {"model_id": -1, "input": [0] * 10}, {"model_id": 200, "input": [0] * 10},
    {"model_id": True, "input": [0] * 10}, {"input": [float("nan")] * 10},
    {"input": [float("inf")] * 10}, {"inputs": [[0] * 10], "actual_output": [0] * 10}])
def test_invalid_predictions(tmp_path, data):
    client = create_app(Config(checkpoints=str(tmp_path), infer_device="cpu"), Queue(1), lambda: True).test_client()
    assert client.post("/predict", json=data, content_type="application/json").status_code == 400


def test_backpressure_and_dead_worker(tmp_path):
    config = Config(infer_device="cpu", checkpoints=str(tmp_path))
    jobs = Queue(1)
    client = create_app(config, jobs, lambda: True).test_client()
    data = {"inputs": [[0] * 10], "outputs": [[0] * 10]}
    assert client.post("/observe", json=data).status_code == 202
    assert client.post("/observe", json=data).status_code == 503
    assert client.post("/observe", json={**data, "outputs": [[0] * 10] * 2}).status_code == 400
    model = MIMOModel()
    torch.save({"weights": model.state_dict(), "version": 1}, checkpoint_path(tmp_path, 0))
    dead = create_app(config, jobs, lambda: False).test_client()
    assert dead.get("/health").status_code == 503
    assert dead.post("/predict", json={"input": [0] * 10}).status_code == 200
    assert dead.post("/predict", json={"input": [0] * 10, "actual_output": [0] * 10}).status_code == 503
    assert dead.post("/observe", json=data).status_code == 503


def test_synthetic_target_matches_original():
    xs = np.random.RandomState(7).uniform(-1, 1, (8, 10))
    a, b, c, d = coefficients(0)
    original = np.array([[np.sum(a[i] * np.sin(b[i] * x**2)) + c[i] * np.cos(np.sum(d[i] * x))
                          for i in range(10)] for x in xs])
    np.testing.assert_allclose(target(xs, (a, b, c, d)), original, atol=1e-14)


def test_explicit_devices(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    validate_devices(Config())
    validate_devices(Config(infer_device="cpu", train_device="cpu"))
    with pytest.raises(ValueError):
        validate_devices(Config(train_device="cuda:0"))
    with pytest.raises(ValueError):
        validate_devices(Config(train_device="cuda:2"))
