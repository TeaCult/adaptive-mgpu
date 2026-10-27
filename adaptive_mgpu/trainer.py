"""Single owner of training models; only plain CPU data crosses the queue."""
from collections import deque
import logging
import os
from pathlib import Path
from queue import Empty

import torch

from .model import MIMOModel


def checkpoint_path(directory, model_id):
    return Path(directory) / f"model-{model_id}.pt"


def atomic_save(payload, path):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train_worker(config, jobs, stop):
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    device = torch.device(config.train_device)
    models = {}
    try:
        while not stop.is_set():
            try:
                model_id, xs, ys, initial = jobs.get(timeout=0.2)
            except Empty:
                continue
            if model_id not in models:
                model = MIMOModel().to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
                version = 0
                path = checkpoint_path(config.checkpoints, model_id)
                if path.exists():
                    state = torch.load(path, map_location=device, weights_only=True)
                    model.load_state_dict(state["weights"])
                    optimizer.load_state_dict(state["optimizer"])
                    version = state["version"]
                models[model_id] = [model, optimizer, deque(maxlen=config.replay_size), version]
            entry = models[model_id]
            model, optimizer, replay, version = entry
            replay.extend(zip(xs, ys))
            # Recent labels are always included. Replay supplies older examples.
            recent_x = torch.tensor(xs, dtype=torch.float32, device=device)
            recent_y = torch.tensor(ys, dtype=torch.float32, device=device)
            replay_x = torch.tensor([x for x, _ in replay], dtype=torch.float32, device=device)
            replay_y = torch.tensor([y for _, y in replay], dtype=torch.float32, device=device)
            model.train()
            steps = config.initial_steps if initial else config.update_steps
            for _ in range(steps):
                if stop.is_set():
                    return
                indices = torch.randint(len(replay), (min(config.batch_size, len(replay)),), device=device)
                indices_new = torch.randint(len(xs), (min(config.batch_size, len(xs)),), device=device)
                bx = torch.cat((recent_x[indices_new], replay_x[indices]))
                by = torch.cat((recent_y[indices_new], replay_y[indices]))
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(model(bx), by)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss for model {model_id}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
                optimizer.step()
            if not all(torch.isfinite(p).all().item() for p in model.parameters()):
                raise RuntimeError(f"Non-finite weights for model {model_id}")
            version += 1
            atomic_save({"weights": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                         "optimizer": optimizer.state_dict(), "version": version},
                        checkpoint_path(config.checkpoints, model_id))
            entry[3] = version
            logging.info("model=%d version=%d loss=%.6f replay=%d", model_id, version, loss.item(), len(replay))
    except Exception:
        logging.exception("Trainer stopped; inspect this error before restarting")
        raise
