"""Lightweight reaction logging for regime-aligned analysis."""

import json
import os
from typing import Iterable, Optional

from regimes import Regime


class ReactionRecorder:
    def __init__(self, path: str, flush_interval: int = 100) -> None:
        self._path = path
        self._flush_interval = max(1, int(flush_interval))
        self._buffer = []
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    def record(self, step: int, regimes: Iterable[Regime], metrics: dict) -> None:
        record = {
            "step": int(step),
            "regimes": [regime.id for regime in regimes],
        }
        record.update(metrics)
        self._buffer.append(record)
        if len(self._buffer) >= self._flush_interval:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        with open(self._path, "a", encoding="utf-8") as f:
            for record in self._buffer:
                f.write(json.dumps(record, sort_keys=True) + "\n")
        self._buffer.clear()

    def close(self) -> None:
        self.flush()
