"""Trusted JevHarness evaluator for Wordle (same pattern as examples/pokemon/adapter.py).

The adapter owns the hidden answer, legality checks and the reward.
The harness only sees build_observation(...).
"""
import copy
import time

from auto_jev.runtime import PipelineExecutionError, PipelineRuntime

from .data import load_words
from .game import FAIL_TURNS, InvalidGuess, WordleGame

ANSWERS, ALLOWED = load_words()
# JevHarness rejects any JSON list longer than 10,000 items, and the full guess
# list is ~13k words, so it is sent as one space-separated string (~78 KB).
# Both lists are public knowledge in real Wordle, so they are fair to observe.
_ANSWER_POOL = list(ANSWERS)
_ALLOWED_TEXT = " ".join(sorted(ALLOWED))
_LEGEND = {"G": "right letter, right position",
           "Y": "letter in the word, wrong position",
           "B": "letter not in the word (or no further copies)"}


def build_observation(game):
    obs = game.observation()
    obs.update(answer_pool=_ANSWER_POOL, allowed_guesses=_ALLOWED_TEXT,
               feedback_legend=_LEGEND)
    return obs


def evaluate_episode(spec, episode, jev, *, capture_traces=True):
    """Score = 7 - turns (solve in 1 -> 6, unsolved or invalid guess -> 0).

    Higher is better, as run_evolution requires; mean turns = 7 - mean score.
    """
    started = time.perf_counter()
    runtime = PipelineRuntime(spec, jev)
    game = WordleGame(episode["answer"], ALLOWED)
    result = {"episode_id": episode["id"], "task_id": "wordle", "status": "running",
              "score": None, "turns": None, "solved": False, "traces": []}
    memory = {}
    try:
        while not game.done:
            obs = build_observation(game)
            light_obs = game.observation()  # constant word lists left out of the record
            before = copy.deepcopy(memory)
            t0 = time.perf_counter()
            try:
                decision = runtime.run(obs, memory)
            except PipelineExecutionError as exc:
                partial = copy.deepcopy(exc.partial_result)
                partial.update(observation=light_obs, status="error")
                result["traces"].append(partial)
                raise
            record = {**decision, "observation": light_obs, "memory_before": before,
                      "decision_ms": (time.perf_counter() - t0) * 1000, "status": "ok"}
            result["traces"].append(record)
            try:
                record["feedback"] = game.guess(decision["output"])
                record["action"] = game.history[-1]["guess"]
            except InvalidGuess as exc:
                # A policy mistake, not infrastructure: the game ends unsolved.
                record.update(status="invalid_action", error=str(exc))
                result.update(status="invalid_action", error=str(exc))
                break
            memory = copy.deepcopy(decision["memory"])

        turns = game.turns_used() if result["status"] == "running" else FAIL_TURNS
        if result["status"] == "running":
            result["status"] = "completed"
        result.update(solved=game.solved and result["status"] == "completed",
                      turns=turns, score=float(FAIL_TURNS - turns),
                      answer=episode["answer"],  # revealed only after the game ends
                      history=copy.deepcopy(game.history),
                      decisions=len(result["traces"]),
                      elapsed_ms=(time.perf_counter() - started) * 1000)
        if not capture_traces:
            result["traces"] = []
        return result
    except BaseException as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {str(exc)[:1000]}",
                      history=copy.deepcopy(game.history),
                      elapsed_ms=(time.perf_counter() - started) * 1000)
        if isinstance(exc, PipelineExecutionError):
            raise PipelineExecutionError(str(exc), partial_result=result,
                                         cause=exc.cause, causes=exc.causes) from exc
        if isinstance(exc, Exception):
            raise PipelineExecutionError("Wordle episode failed",
                                         partial_result=result, cause=exc) from exc
        raise


def evaluate_code_episode(spec, episode, jev, *, capture_traces=True):
    """Ablation: the same harness format, but no Jev nodes allowed."""
    if any(node.get("kind") == "jev" for node in spec.get("nodes", [])):
        raise ValueError("The code-only ablation prohibits Jev nodes")
    return evaluate_episode(spec, episode, jev, capture_traces=capture_traces)
