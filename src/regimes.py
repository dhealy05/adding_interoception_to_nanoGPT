"""Regime definitions and legacy compatibility helpers."""

from dataclasses import dataclass, field
import json
import os
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class Regime:
    id: str
    start: int
    end: int
    intensity: float = 1.0
    channels: Tuple[str, ...] = ("target",)
    params: Dict[str, object] = field(default_factory=dict)

    def is_active(self, step: int) -> bool:
        return self.start <= step < self.end


class RegimeManager:
    def __init__(self, regimes: Iterable[Regime]):
        self._regimes = list(regimes)

    def active(self, step: int) -> List[Regime]:
        return [regime for regime in self._regimes if regime.is_active(step)]


def build_legacy_bewilderment_regimes(
    enabled: bool,
    start: int,
    end: int,
    prob: float,
) -> List[Regime]:
    if not enabled:
        return []
    return [
        Regime(
            id="bewilderment",
            start=start,
            end=end,
            intensity=prob,
            channels=("target",),
            params={"mode": "random_targets"},
        )
    ]


def _coerce_channels(channels) -> Tuple[str, ...]:
    if channels is None:
        return ("target",)
    if isinstance(channels, str):
        return (channels,)
    return tuple(channels)


def load_regimes_from_path(path: str) -> List[Regime]:
    if not path:
        return []
    if not os.path.isfile(path):
        raise FileNotFoundError(f"regime config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "regimes" in data:
        items = data["regimes"]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError("regime config must be a list or a dict with 'regimes'")
    regimes: List[Regime] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"regime entry {idx} must be an object")
        if item.get("enabled", True) is False:
            continue
        if "start" not in item or "end" not in item:
            raise ValueError(f"regime entry {idx} missing start/end")
        start = int(item["start"])
        end = int(item["end"])
        if end <= start:
            raise ValueError(f"regime entry {idx} has end <= start")
        rid = item.get("id") or f"regime_{idx}"
        if "intensity" in item:
            intensity = float(item["intensity"])
        elif "prob" in item:
            intensity = float(item["prob"])
        else:
            intensity = 1.0
        channels = _coerce_channels(item.get("channels"))
        params = item.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ValueError(f"regime entry {idx} params must be an object")
        regimes.append(
            Regime(
                id=str(rid),
                start=start,
                end=end,
                intensity=intensity,
                channels=channels,
                params=params,
            )
        )
    return regimes
