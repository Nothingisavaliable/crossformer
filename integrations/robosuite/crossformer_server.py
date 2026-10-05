"""HTTP inference server for the official CrossFormer Bridge policy."""

from __future__ import annotations

import argparse
from collections import deque
import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
import jax

# Initialize the CUDA 13 PJRT plugin before TensorFlow loads its own XLA/CUDA
# symbols. TensorFlow is used only for checkpoint/text utilities and is hidden
# from the GPU below.
JAX_DEVICES = tuple(jax.devices())

import json_numpy
import numpy as np
from PIL import Image
import tensorflow as tf
import uvicorn

from crossformer.model.crossformer_model import CrossFormerModel


# FastAPI / Starlette parses request bodies through the standard json module.
# Patching it lets numpy arrays encoded by the robosuite client be restored
# before endpoint validation runs.
json_numpy.patch()
tf.config.set_visible_devices([], "GPU")


CHECKPOINT = "hf://rail-berkeley/crossformer"
HEAD_NAME = "single_arm"
DATASET_NAME = "bridge_dataset"
HISTORY_LENGTH = 5
IMAGE_SIZE = (224, 224)


def _json_response(value: Any) -> Response:
    return Response(content=json_numpy.dumps(value), media_type="application/json")


def _resize_rgb(image: Any) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected an HxWx3 RGB image, got {image.shape}")
    image = np.clip(image, 0, 255).astype(np.uint8, copy=False)
    return np.asarray(Image.fromarray(image).resize(IMAGE_SIZE, Image.Resampling.LANCZOS))


class BridgePolicyServer:
    def __init__(self) -> None:
        self.model = CrossFormerModel.load_pretrained(CHECKPOINT)
        if DATASET_NAME not in self.model.dataset_statistics:
            raise KeyError(f"Checkpoint has no statistics for {DATASET_NAME!r}")
        self.action_statistics = self.model.dataset_statistics[DATASET_NAME]["action"]
        self.history: deque[np.ndarray] = deque(maxlen=HISTORY_LENGTH)
        self.num_observations = 0
        self.task = None
        self.rng = jax.random.PRNGKey(0)
        self.lock = threading.Lock()

    def reset(self, instruction: str) -> None:
        if not instruction:
            raise ValueError("instruction must be non-empty")
        with self.lock:
            self.task = self.model.create_tasks(texts=[instruction])
            self.history.clear()
            self.num_observations = 0
            self.rng = jax.random.PRNGKey(0)

    def predict(self, image: Any) -> np.ndarray:
        with self.lock:
            if self.task is None:
                raise ValueError("Call /reset with an instruction before /query")

            image = _resize_rgb(image)
            if not self.history:
                self.history.extend([image] * HISTORY_LENGTH)
            else:
                self.history.append(image)
            self.num_observations += 1
            images = np.stack(tuple(self.history), axis=0)[None]
            valid_count = min(self.num_observations, HISTORY_LENGTH)
            timestep_pad_mask = np.zeros((1, HISTORY_LENGTH), dtype=bool)
            timestep_pad_mask[:, -valid_count:] = True
            observation = {
                "image_primary": images,
                "timestep_pad_mask": timestep_pad_mask,
            }

            self.rng, sample_key = jax.random.split(self.rng)
            action_chunk = self.model.sample_actions(
                observation,
                self.task,
                head_name=HEAD_NAME,
                unnormalization_statistics=self.action_statistics,
                rng=sample_key,
                train=False,
            )
            # Receding-horizon control: execute the first action and replan from
            # the next robosuite observation.
            action = np.asarray(jax.device_get(action_chunk))[0, 0, :7]
            if action.shape != (7,):
                raise ValueError(f"Expected a 7-D Bridge action, got {action.shape}")
            return action.astype(np.float32)


def create_app() -> FastAPI:
    policy = BridgePolicyServer()
    app = FastAPI(title="CrossFormer Bridge policy for robosuite")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "checkpoint": CHECKPOINT,
            "head_name": HEAD_NAME,
            "dataset_name": DATASET_NAME,
            "devices": [str(device) for device in JAX_DEVICES],
        }

    @app.post("/reset")
    def reset(payload: dict[str, Any]) -> dict[str, str]:
        try:
            policy.reset(str(payload.get("instruction", "")))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": "reset"}

    @app.post("/query")
    def query(payload: dict[str, Any]) -> Response:
        try:
            action = policy.predict(payload["image_primary"])
        except KeyError as exc:
            raise HTTPException(status_code=400, detail="Missing image_primary") from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _json_response(action)

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
