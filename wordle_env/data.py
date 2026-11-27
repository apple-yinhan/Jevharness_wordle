"""Word lists and the train / selection / test split."""
from pathlib import Path
import json
import random
import re

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_WORD = re.compile(r"^[a-z]{5}$")


def _read(path):
    words = [w.strip().lower() for w in Path(path).read_text().split()]
    bad = [w for w in words if not _WORD.match(w)]
    if bad:
        raise ValueError(f"{path}: not 5-letter words: {bad[:5]}")
    return words


def load_words(data_dir=DATA_DIR):
    """answers.txt = possible hidden words; allowed.txt = extra legal guesses."""
    answers = sorted(set(_read(Path(data_dir) / "answers.txt")))
    allowed = frozenset(_read(Path(data_dir) / "allowed.txt")) | frozenset(answers)
    return answers, allowed


def make_splits(answers, seed=0, fractions=(0.70, 0.15, 0.15)):
    """Fixed, seeded split. Episode IDs do not contain the answer."""
    words = sorted(answers)
    random.Random(seed).shuffle(words)
    n_train = round(len(words) * fractions[0])
    n_select = round(len(words) * fractions[1])
    parts = {"train": words[:n_train],
             "select": words[n_train:n_train + n_select],
             "test": words[n_train + n_select:]}
    return {name: [{"id": f"{name}-{i:04d}", "answer": w} for i, w in enumerate(ws)]
            for name, ws in parts.items()}


if __name__ == "__main__":
    answers, allowed = load_words()
    splits = make_splits(answers)
    out = DATA_DIR / "splits.json"
    out.write_text(json.dumps(splits, indent=1))
    print({k: len(v) for k, v in splits.items()}, f"allowed={len(allowed)}", "->", out)
