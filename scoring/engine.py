"""Standalone Telemanom-ESA inference: load lab artifacts, score new telemetry."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Union

import numpy as np
import pandas as pd

from .errors import Errors
from .windows import PredictionWindows

PathLike = Union[str, os.PathLike]


def default_artifacts_dir() -> Path:
    """Default: esa-telemanom-agent/model/ (lab weights + contract)."""
    return Path(__file__).resolve().parents[1] / "model"


def _as_config(contract: dict) -> SimpleNamespace:
    cfg = SimpleNamespace(**contract)
    cfg.train = False
    cfg.predict = True
    return cfg


@dataclass
class InferenceResult:
    timestamps: pd.Index
    scores: pd.DataFrame
    alarms: Optional[pd.DataFrame]


class TelemanomInferencer:
    """Load a trained Telemanom-ESA model and run inference on a telemetry CSV."""

    def __init__(self, artifacts_dir: Optional[PathLike] = None):
        self.artifacts_dir = Path(artifacts_dir or default_artifacts_dir()).resolve()
        self.contract = json.loads((self.artifacts_dir / "contract.json").read_text())
        self.config = _as_config(self.contract)
        self.means = np.atleast_1d(np.loadtxt(self.artifacts_dir / "model.h5.means"))
        self.stds = np.atleast_1d(np.loadtxt(self.artifacts_dir / "model.h5.stds"))
        n = len(self.contract["input_channels"])
        if self.means.shape[0] != n or self.stds.shape[0] != n:
            raise ValueError(
                f"means/stds length {self.means.shape[0]}/{self.stds.shape[0]} "
                f"does not match {n} input channels"
            )
        self.model = None

    def load_model(self):
        if self.model is not None:
            return self.model
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf
        from tensorflow.keras.models import load_model

        for device in tf.config.experimental.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(device, True)
            except Exception:
                pass
        path = self.artifacts_dir / "model.h5"
        self.model = load_model(path, compile=False)
        return self.model

    def load_telemetry(
        self,
        csv_path: PathLike,
        max_rows: Optional[int] = None,
        skip_rows: int = 0,
    ) -> pd.DataFrame:
        path = Path(csv_path)
        header = pd.read_csv(path, nrows=0)
        columns = header.columns.tolist()
        if "timestamp" not in columns:
            raise ValueError(f"{path} has no 'timestamp' column")

        needed = list(self.contract["input_channels"])
        missing = [c for c in needed if c not in columns]
        if missing:
            raise ValueError(
                f"{path} is missing {len(missing)} required input columns, e.g. {missing[:8]}"
            )

        usecols = ["timestamp"] + needed
        # Keep header row; skip the first skip_rows data rows.
        skip = range(1, skip_rows + 1) if skip_rows > 0 else None
        df = pd.read_csv(
            path,
            usecols=usecols,
            index_col="timestamp",
            parse_dates=True,
            dtype={c: np.float32 for c in needed},
            skiprows=skip,
            nrows=max_rows,
        )
        return df.loc[:, needed]

    def _windows(self, values: np.ndarray) -> PredictionWindows:
        arr = np.expand_dims(np.asarray(values, dtype=np.float32), axis=0)
        windows = PredictionWindows(
            arr,
            self.means,
            self.stds,
            self.contract["input_channel_indices"],
            self.contract["target_channel_indices"],
            window_size=self.contract["window_size"],
            prediction_window_size=self.contract["prediction_window_size"],
            batch_size=self.contract["batch_size"],
        )
        if windows.nb_samples < 1:
            raise ValueError(
                f"Need more than window_size={self.contract['window_size']} timesteps; "
                f"got {values.shape[0]}"
            )
        return windows

    def _predict_yhat(self, windows: PredictionWindows) -> np.ndarray:
        model = self.load_model()
        n_samples = windows.nb_samples
        n_channels = len(self.contract["target_channel_indices"])
        y_hat = np.empty((n_samples, n_channels), dtype=np.float32)
        cursor = 0
        n_batches = len(windows)
        log_every = max(1, n_batches // 20)
        for i in range(n_batches):
            if i % log_every == 0:
                print(
                    f"Predict batch {i}/{n_batches} ({100.0 * i / max(n_batches, 1):.0f}%)",
                    flush=True,
                )
            batch = model(windows[i], training=False).numpy()[:, 0, :]
            n = int(batch.shape[0])
            y_hat[cursor : cursor + n] = batch.astype(np.float32, copy=False)
            cursor += n
        y_hat = y_hat[:cursor]
        tgt_idx = self.contract["target_channel_indices"]
        y_hat = y_hat * windows.train_stds[tgt_idx] + windows.train_means[tgt_idx]
        print(f"Prediction complete: {cursor} samples", flush=True)
        return y_hat

    def predict(
        self,
        csv_path: PathLike,
        threshold: bool = True,
        max_rows: Optional[int] = None,
        skip_rows: int = 0,
    ) -> InferenceResult:
        frame = self.load_telemetry(csv_path, max_rows=max_rows, skip_rows=skip_rows)
        window_size = int(self.contract["window_size"])
        batch_size = int(self.contract["batch_size"])
        # Telemanom dynamic thresholding needs history of ~smoothing_window_size * batch_size
        # after the look-back window. Recommend at least window + 2100 rows.
        min_recommended = window_size + batch_size * int(self.contract["smoothing_window_size"])
        if threshold and len(frame) < min_recommended:
            print(
                f"Warning: only {len(frame)} rows (need ~{min_recommended}+ for stable "
                f"Telemanom thresholding). Prefer --max-rows {min_recommended} or higher.",
                flush=True,
            )
        windows = self._windows(frame.values)
        y_hat = self._predict_yhat(windows)

        channel = SimpleNamespace(generator_test=windows, id=self.contract["target_channels"])
        model = SimpleNamespace(y_hat=y_hat)
        errors = Errors(channel, model, self.config, run_id="infer", save=False)

        n_out = y_hat.shape[1]
        padded_scores = np.concatenate(
            (np.zeros((window_size, n_out), dtype=np.float32), errors.e_s.astype(np.float32)),
            axis=0,
        )
        if padded_scores.shape[0] < len(frame):
            pad = np.zeros((len(frame) - padded_scores.shape[0], n_out), dtype=np.float32)
            padded_scores = np.concatenate((padded_scores, pad), axis=0)
        elif padded_scores.shape[0] > len(frame):
            padded_scores = padded_scores[: len(frame)]

        targets = list(self.contract["target_channels"])
        scores = pd.DataFrame(padded_scores, index=frame.index, columns=targets)

        alarms = None
        if threshold:
            thresholded = np.zeros_like(y_hat, dtype=np.float32)
            for channel_idx in self.contract["target_channel_indices"]:
                errors.process_batches(channel, channel_idx)
                col = self.contract["target_channel_indices"].index(channel_idx)
                for start_idx, end_idx in errors.E_seq:
                    thresholded[start_idx:end_idx, col] = 1
            padded_alarms = np.concatenate(
                (np.zeros((window_size, n_out), dtype=np.float32), thresholded),
                axis=0,
            )
            if padded_alarms.shape[0] < len(frame):
                pad = np.zeros((len(frame) - padded_alarms.shape[0], n_out), dtype=np.float32)
                padded_alarms = np.concatenate((padded_alarms, pad), axis=0)
            elif padded_alarms.shape[0] > len(frame):
                padded_alarms = padded_alarms[: len(frame)]
            alarms = pd.DataFrame(padded_alarms, index=frame.index, columns=targets)

        return InferenceResult(timestamps=frame.index, scores=scores, alarms=alarms)


def write_outputs(result: InferenceResult, output_dir: PathLike) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.scores.to_csv(out / "scores.csv")
    if result.alarms is not None:
        result.alarms.to_csv(out / "alarms.csv")
    print(f"Wrote {out / 'scores.csv'}")
    if result.alarms is not None:
        print(f"Wrote {out / 'alarms.csv'}")
