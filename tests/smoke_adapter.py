"""End-to-end smoke test of the trusted adapter with a code-only harness.

Makes no Jev calls (the harness has no Jev nodes). Needs the macOS sandbox.
Run: .venv/bin/python -m tests.smoke_adapter
"""
import json
from pathlib import Path

from wordle_env.adapter import evaluate_code_episode

ROOT = Path(__file__).resolve().parent.parent
spec = json.loads((ROOT / "harnesses" / "code_only_first_candidate.json").read_text())
episodes = json.loads((ROOT / "data" / "splits.json").read_text())["train"][:5]
for ep in episodes:
    r = evaluate_code_episode(spec, ep, jev=None)
    print(ep["id"], r["status"], r["answer"], r["turns"], r["score"],
          [h["guess"] + ":" + h["feedback"] for h in r["history"]],
          [round(t["decision_ms"]) for t in r["traces"]], "ms")
