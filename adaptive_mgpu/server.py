"""One inference process and one spawned training process, on explicit devices."""
import argparse
from dataclasses import dataclass
import logging
import multiprocessing as mp
from pathlib import Path
from queue import Full
from threading import Lock

from flask import Flask, jsonify, request
import numpy as np
import torch
from werkzeug.exceptions import BadRequest

from .model import MIMOModel
from .trainer import checkpoint_path, train_worker


@dataclass
class Config:
    infer_device: str = "cuda:0"
    train_device: str = "cuda:1"
    instances: int = 200
    checkpoints: str = "checkpoints"
    queue_size: int = 256
    replay_size: int = 2048
    batch_size: int = 64
    initial_steps: int = 500
    update_steps: int = 10
    learning_rate: float = 0.001
    seed: int = 0


def matrix(value, name, limit):
    try:
        result = np.asarray(value, dtype=np.float32)
    except (ValueError, TypeError, OverflowError):
        raise ValueError(f"{name} must be a numeric matrix") from None
    if result.ndim != 2 or result.shape[1] != 10 or not 1 <= len(result) <= limit:
        raise ValueError(f"{name} must have shape [1..{limit}, 10]")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite float32 values")
    return result


def create_app(config, jobs, trainer_alive):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024
    models = {}
    lock = Lock()
    device = torch.device(config.infer_device)

    @app.errorhandler(ValueError)
    @app.errorhandler(BadRequest)
    def invalid(error):
        return jsonify(error=str(error)), 400

    def payload():
        data = request.get_json()
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object")
        model_id = data.get("model_id", 0)
        if type(model_id) is not int or not 0 <= model_id < config.instances:
            raise ValueError(f"model_id must be an integer in [0, {config.instances - 1}]")
        return data, model_id

    def enqueue(model_id, xs, ys, initial):
        if not trainer_alive():
            return jsonify(error="Trainer unavailable; labels were not accepted"), 503
        try:
            jobs.put_nowait((model_id, xs.tolist(), ys.tolist(), initial))
        except Full:
            return jsonify(error="Training queue full; retry labels later"), 503
        return None

    @app.get("/health")
    def health():
        alive = trainer_alive()
        with lock:
            versions = {str(i): v for i, (_, _, v) in models.items()}
        return jsonify(trainer_alive=alive, infer_device=str(device),
                       train_device=config.train_device, instances=config.instances,
                       loaded_versions=versions), 200 if alive else 503

    @app.post("/train")
    @app.post("/observe")
    def observe():
        data, model_id = payload()
        xs = matrix(data.get("inputs"), "inputs", config.replay_size)
        ys = matrix(data.get("outputs"), "outputs", config.replay_size)
        if len(xs) != len(ys):
            raise ValueError("inputs and outputs must have equal row counts")
        rejected = enqueue(model_id, xs, ys, request.path == "/train")
        if rejected is not None:
            return rejected
        return jsonify(status="queued", model_id=model_id, samples=len(xs)), 202

    @app.post("/predict")
    def predict():
        data, model_id = payload()
        single = "input" in data
        xs = matrix([data["input"]] if single else data.get("inputs"), "inputs", 1024)
        ys = None
        if "actual_output" in data:
            if not single:
                raise ValueError("Use /observe for batched or delayed labels")
            ys = matrix([data["actual_output"]], "actual_output", 1)
        with lock:
            path = checkpoint_path(config.checkpoints, model_id)
            if not path.exists():
                return jsonify(error="Model untrained; POST /train and retry after publication"), 503
            stamp = path.stat().st_mtime_ns
            if model_id not in models or models[model_id][0] != stamp:
                state = torch.load(path, map_location="cpu", weights_only=True)
                candidate = MIMOModel().to(device)
                candidate.load_state_dict(state["weights"])
                candidate.eval()
                models[model_id] = (stamp, candidate, state["version"])
            _, model, version = models[model_id]
            with torch.inference_mode():
                result = model(torch.as_tensor(xs, device=device)).cpu().tolist()
        if ys is not None:
            rejected = enqueue(model_id, xs, ys, False)
            if rejected is not None:
                return rejected
        return jsonify(prediction=result[0] if single else result, model_id=model_id,
                       version=version, labels_queued=ys is not None)

    return app


def validate_devices(config):
    devices = [torch.device(config.infer_device), torch.device(config.train_device)]
    for device in devices:
        if device.type not in ("cpu", "cuda"):
            raise ValueError("Only cpu and explicit cuda:N devices are supported")
        if device.type == "cuda" and (device.index is None or device.index >= torch.cuda.device_count()):
            raise ValueError(f"{device} unavailable; check CUDA_VISIBLE_DEVICES and PyTorch CUDA installation")
    if devices[0].type == "cuda" and devices[0] == devices[1]:
        raise ValueError("Use different GPUs for training and inference, or cpu/cpu for a smoke test")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, field in Config.__dataclass_fields__.items():
        parser.add_argument("--" + name.replace("_", "-"), type=field.type, default=field.default)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    config = Config(**{key: getattr(args, key) for key in Config.__dataclass_fields__})
    for name in ("instances", "queue_size", "replay_size", "batch_size", "initial_steps", "update_steps"):
        if getattr(config, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 < config.learning_rate < 1:
        parser.error("--learning-rate must be between 0 and 1")
    validate_devices(config)
    Path(config.checkpoints).mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    logging.basicConfig(level=logging.INFO)
    context = mp.get_context("spawn")
    jobs = context.Queue(maxsize=config.queue_size)
    stop = context.Event()
    trainer = context.Process(target=train_worker, args=(config, jobs, stop), name="trainer")
    trainer.start()
    try:
        app = create_app(config, jobs, trainer.is_alive)
        # Single process owns the inference bank. Do not add WSGI worker processes.
        app.run(host=args.host, port=args.port, debug=False, use_reloader=False, threaded=True)
    finally:
        stop.set()
        trainer.join(timeout=10)
        if trainer.is_alive():
            trainer.terminate()
            trainer.join(timeout=5)
        jobs.close()
        jobs.cancel_join_thread()


if __name__ == "__main__":
    main()
