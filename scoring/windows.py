"""Sliding-window batches for Telemanom-ESA inference (prediction mode only)."""

from __future__ import annotations

import numpy as np


class PredictionWindows:
    """Same indexing as TelemanomGenerator(..., prediction_mode=True)."""

    def __init__(
        self,
        data: np.ndarray,
        train_means: np.ndarray,
        train_stds: np.ndarray,
        input_channel_indices,
        target_channel_indices,
        window_size: int = 250,
        prediction_window_size: int = 10,
        batch_size: int = 70,
    ):
        self.data = data
        self.train_means = train_means
        self.train_stds = train_stds
        self.input_channel_indices = list(input_channel_indices)
        self.target_channel_indices = list(target_channel_indices)
        self.window_size = int(window_size)
        self.prediction_window_size = int(prediction_window_size)
        self.batch_size = int(batch_size)

        frag_ids = []
        offsets = []
        for i, arr in enumerate(self.data):
            for j in range(len(arr) - self.window_size):
                frag_ids.append(i)
                offsets.append(j)
        self.frag_ids = np.asarray(frag_ids, dtype=np.int32)
        self.offsets = np.asarray(offsets, dtype=np.int32)
        self.nb_samples = int(self.frag_ids.shape[0])
        self._order = np.arange(self.nb_samples, dtype=np.int64)

    def __len__(self) -> int:
        if self.nb_samples == 0:
            return 0
        return int(np.ceil(self.nb_samples / self.batch_size))

    def __getitem__(self, index: int) -> np.ndarray:
        start = index * self.batch_size
        end = min(start + self.batch_size, self.nb_samples)
        order = self._order[start:end]
        n = len(order)
        n_in = len(self.input_channel_indices)
        means_in = self.train_means[self.input_channel_indices]
        stds_in = self.train_stds[self.input_channel_indices]
        input_data = np.empty((n, self.window_size, n_in), dtype=np.float32)
        for k, idx in enumerate(order):
            i = int(self.frag_ids[idx])
            j = int(self.offsets[idx])
            window_end = j + self.window_size
            input_data[k] = (
                self.data[i][j:window_end, self.input_channel_indices] - means_in
            ) / stds_in
        return input_data
