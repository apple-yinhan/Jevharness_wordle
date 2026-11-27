"""Rule-based baseline: replay a precomputed optimal Wordle decision tree.

Tree source: Wordle-master/solutions/*.pp.txt (gardenofminds "Adventures in
Wordle"), format (guess, {feedback_lowercase_gby: subtree_or_final_word}).
Parsed with ast.literal_eval, never unpickled.
"""
import ast
import json
import sys
from pathlib import Path
from statistics import mean

from wordle_env.data import load_words
from wordle_env.game import play_episode

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TREE = ROOT / "Wordle-master" / "solutions" / "official_goal_solution.pp.txt"


def load_tree(path=DEFAULT_TREE):
    return ast.literal_eval(Path(path).read_text())


def make_policy(tree):
    def policy(obs):
        node = tree
        for step in obs["history"]:
            node = node[1][step["feedback"].lower()]
        return node if isinstance(node, str) else node[0]
    return policy


def summarize(results):
    return {"games": len(results),
            "solved": sum(r["solved"] for r in results),
            "mean_turns": mean(r["turns"] for r in results),
            "max_turns": max(r["turns"] for r in results),
            "mean_decision_ms": mean(m for r in results for m in r["decision_ms"])}


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "all"
    answers, allowed = load_words()
    if split != "all":
        splits = json.loads((ROOT / "data" / "splits.json").read_text())
        answers = [e["answer"] for e in splits[split]]
    policy = make_policy(load_tree())
    results = [play_episode(policy, a, allowed) for a in answers]
    print(split, summarize(results))
