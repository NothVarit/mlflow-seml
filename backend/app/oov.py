import csv
import json
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import date, timezone, datetime
from pathlib import Path

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_lock = threading.Lock()


def _load_vocab(tokenizer_json_path: str) -> frozenset[str]:
    with open(tokenizer_json_path, encoding="utf-8") as f:
        tok = json.load(f)
    raw = tok.get("model", {}).get("vocab", {})
    # WordPiece vocab keys — strip ## prefix to get surface tokens
    tokens = set()
    for key in raw:
        tokens.add(key.lstrip("#"))
    return frozenset(tokens)


def oov_rate(text: str, vocab: frozenset[str]) -> float:
    tokens = TOKEN_PATTERN.findall(text.lower())
    if not tokens:
        return 0.0
    unseen = sum(1 for t in tokens if t not in vocab)
    return unseen / len(tokens)


def append_oov_history(history_path: str, rate: float) -> None:
    today = date.today().isoformat()
    with _lock:
        path = Path(history_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or path.stat().st_size == 0
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["date", "oov_rate"])
            writer.writerow([today, f"{rate:.6f}"])


def spike_detected(history_path: str, current_rate: float, sigma_threshold: float = 2.0) -> bool:
    """Return True if current_rate exceeds mean + sigma_threshold * std of recent history."""
    import numpy as np

    path = Path(history_path)
    if not path.exists():
        return False
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rates = []
        for row in reader:
            try:
                rates.append(float(row.get("oov_rate") or row.get("unseen_token_rate") or row.get("unseen_word_rate")))
            except (TypeError, ValueError):
                continue
    if len(rates) < 2:
        return False
    arr = np.array(rates[-14:])  # last 14 days window
    threshold = arr.mean() + sigma_threshold * arr.std(ddof=0)
    return current_rate > threshold


@dataclass
class OovDetector:
    vocab: frozenset[str]
    history_path: str
    sigma_threshold: float = 2.0
    enabled: bool = True

    @classmethod
    def from_env(cls) -> "OovDetector":
        tokenizer_path = os.getenv("TOKENIZER_JSON_PATH", "")
        history_path = os.getenv("OOV_HISTORY_PATH", "")
        sigma = float(os.getenv("OOV_SIGMA_THRESHOLD", "2.0"))
        if not tokenizer_path or not history_path:
            return cls(vocab=frozenset(), history_path="", enabled=False)
        try:
            vocab = _load_vocab(tokenizer_path)
            return cls(vocab=vocab, history_path=history_path, sigma_threshold=sigma, enabled=True)
        except Exception:
            return cls(vocab=frozenset(), history_path="", enabled=False)

    def check(self, text: str) -> tuple[float, bool]:
        """Returns (oov_rate, spike_detected). Appends to history and checks threshold."""
        if not self.enabled:
            return 0.0, False
        rate = oov_rate(text, self.vocab)
        append_oov_history(self.history_path, rate)
        triggered = spike_detected(self.history_path, rate, self.sigma_threshold)
        return rate, triggered
