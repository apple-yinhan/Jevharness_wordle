# JevHarness for Wordle

## Layout

```
wordle_env/        game rules, word lists + splits, trusted JevHarness adapter
baselines/         rule-based baselines (no models)
  entropy_solver.py   greedy max-entropy solver  -> strong rule-based baseline
  optimal_tree.py     replays a Wordle-master decision tree (NOT mean-optimal, see below)
harnesses/         PipelineSpec JSON harnesses
tests/             pytest unit tests + adapter smoke test
data/              answers.txt (2315), allowed.txt (12972), splits.json, feedback_matrix.npy (cache)
third_party/JevHarness   JevHarness checkout (commit d767603), installed editable in .venv
```

## Setup

```bash
/opt/homebrew/bin/python3.12 -m venv .venv          # JevHarness needs Python >= 3.11
.venv/bin/pip install -e "third_party/JevHarness[dev]" numpy
.venv/bin/python -m wordle_env.data                  # -> data/splits.json (70/15/15, seed 0)
.venv/bin/python -m pytest -q tests                  # rule unit tests
.venv/bin/python -m tests.smoke_adapter              # adapter end-to-end, code-only harness, no Jev calls
```

Jev and reflection need keys in a local `.env` (never commit it):
`AI_GATEWAY_API_KEY` (Jev via Vercel) or `TYPESAFE_API_KEY` (Jev direct), plus a key for the authoring/reflection LLM.

## Baselines (all 2315 answers)

| Method | mean turns | solved | max |
|---|---|---|---|
| Greedy entropy, opener `salet` (`python -m baselines.entropy_solver [all|train|select|test]`) | 3.433 | 100% | 6 |
| Wordle-master `official_goal_solution` tree | 4.025 | 100% | 5 |
| Wordle-master `official_goal_solution_hard` tree | 3.678 | 100% | 6 |
| Olson (2022) optimum, for reference | 3.421 | 100% | – |

The Wordle-master trees guarantee a win but do not minimise mean turns, so they are not an upper bound.

## Progress 20261127

The engineering setup is done. The code is now a `wordle_env/` package, JevHarness is installed in a Python 3.12 virtual environment, and the data is split 70/15/15 into train / select / test (1620 / 347 / 348 words). All unit tests pass. A code-only harness, with no Jev calls, played 5 complete games through the adapter, so nothing has been spent on API calls yet.

We found two things about rule-based baselines. First, the decision trees bundled with the downloaded Wordle-master repo always win but are not built to minimise average turns: they average 4.02 turns, so they can't serve as an upper bound. Second, our greedy entropy solver averages 3.433 turns with a 100% solve rate when the answer list is known (setting A). That is close to Olson's optimum of 3.421. In setting A every decision can be computed exactly, so Jev has little room to add anything.

So we added setting B, where the harness doesn't know the answer list and only sees the 12,972 legal words. On the test set:

| Solver | Mean turns |
|---|---|
| Entropy solver, uniform prior (every legal word equally likely) | 4.011 |
| Entropy solver, word-frequency prior (from the `wordfreq` library) | 3.649 |
| Entropy solver, answer list known (setting A) | 3.431 |

The frequency prior closes about 62% of the gap. The remaining ~0.22 turns is where common-sense judgment matters. Examples are traps like `_aunt`, where many words differ by one letter, and the fact that Wordle answers are rarely plurals.

The Jev harness now has a clear target in setting B: it must average below 3.649 to show that Jev adds something code alone can't. Possible next steps:
- widen the tuning grid to make the frequency baseline more solid;
- build the LLM-every-turn baseline;
- start the first Jev harness.

Before any real Jev calls, API keys need to go into `.env`.
