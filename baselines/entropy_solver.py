"""Rule-based baseline: greedy max-entropy solver over the full guess list.

pool="answers" (setting A): candidates are the 2315 official answers.
pool="allowed" (setting B): the answer list is unknown, so every one of the
12972 legal words is a candidate with a uniform prior.

Each turn picks the allowed guess whose feedback partition of the remaining
candidates has the highest entropy (ties -> prefer a possible answer, then
smaller worst-case bucket). With <=2 candidates it just guesses a candidate.
The 12972 x 2315 feedback matrix is built once with numpy and cached.
"""
import json
import sys
from pathlib import Path
from statistics import mean

import numpy as np

from wordle_env.data import DATA_DIR, load_words
from wordle_env.game import play_episode, score_guess
from baselines.optimal_tree import ROOT, summarize

CACHE = {"answers": DATA_DIR / "feedback_matrix.npy",
         "allowed": DATA_DIR / "feedback_matrix_allowed.npy"}
OPENER = {"answers": "salet"}  # setting B's opener is computed on first use
CHUNK = 1500
_CODE = {"B": 0, "Y": 1, "G": 2}


def feedback_code(fb):
    return sum(_CODE[c] * 3 ** i for i, c in enumerate(fb))


def build_matrix(guesses, answers):
    """M[g, a] = base-3 feedback code, same duplicate-letter rules as score_guess."""
    G = np.array([[ord(c) - 97 for c in w] for w in guesses], dtype=np.int8)
    M = np.empty((len(guesses), len(answers)), dtype=np.uint8)
    rows = np.arange(len(guesses))
    pow3 = 3 ** np.arange(5)
    for j, a in enumerate(answers):
        av = np.array([ord(c) - 97 for c in a], dtype=np.int8)
        green = G == av
        cnt = np.zeros((len(guesses), 26), dtype=np.int8)
        for p in range(5):
            cnt[:, av[p]] += ~green[:, p]
        res = green.astype(np.int64) * 2
        for p in range(5):
            yel = ~green[:, p] & (cnt[rows, G[:, p]] > 0)
            cnt[rows, G[:, p]] -= yel
            res[:, p] += yel
        M[:, j] = res @ pow3
    return M


class EntropySolver:
    def __init__(self, pool="answers", opener=None):
        answers, allowed = load_words()
        self.guesses = sorted(allowed)
        # self.answers = the candidate pool (matrix columns), not the true answer list
        self.answers = answers if pool == "answers" else self.guesses
        self.gidx = {w: i for i, w in enumerate(self.guesses)}
        cache = CACHE[pool]
        if cache.exists():
            self.M = np.load(cache)
        else:
            self.M = build_matrix(self.guesses, self.answers)
            np.save(cache, self.M)
        self._memo = {}
        self.opener = opener or OPENER.get(pool) or self.best_guess(np.arange(len(self.answers)))

    def candidates(self, history):
        cand = np.arange(len(self.answers))
        for h in history:
            cand = cand[self.M[self.gidx[h["guess"]], cand] == feedback_code(h["feedback"])]
        return cand

    def best_guess(self, cand):
        if len(cand) <= 2:
            return self.answers[cand[0]]
        ng = len(self.guesses)
        counts = np.empty((ng, 243), dtype=np.int64)
        for lo in range(0, ng, CHUNK):  # chunked to bound memory when cand is large
            sub = self.M[lo:lo + CHUNK][:, cand].astype(np.int64)
            k = len(sub)
            flat = (np.arange(k)[:, None] * 243 + sub).ravel()
            counts[lo:lo + k] = np.bincount(flat, minlength=k * 243).reshape(k, 243)
        p = counts / len(cand)
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -np.nansum(np.where(p > 0, p * np.log2(p), 0.0), axis=1)
        in_cand = np.zeros(len(self.guesses), bool)
        in_cand[[self.gidx[self.answers[c]] for c in cand]] = True
        key = np.lexsort((counts.max(axis=1), ~in_cand, -np.round(ent, 9)))
        return self.guesses[key[0]]

    def __call__(self, obs):
        hist = obs["history"]
        if not hist:
            return self.opener
        k = tuple((h["guess"], h["feedback"]) for h in hist)
        if k not in self._memo:
            self._memo[k] = self.best_guess(self.candidates(hist))
        return self._memo[k]


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "all"
    pool = sys.argv[2] if len(sys.argv) > 2 else "answers"
    solver = EntropySolver(pool)
    print("pool:", pool, "opener:", solver.opener)
    answers, allowed = load_words()
    # sanity: matrix agrees with the reference scorer
    rng = np.random.default_rng(0)
    for gi, ai in zip(rng.integers(0, len(solver.guesses), 20000), rng.integers(0, len(solver.answers), 20000)):
        assert solver.M[gi, ai] == feedback_code(score_guess(solver.guesses[gi], solver.answers[ai]))
    if split != "all":
        splits = json.loads((ROOT / "data" / "splits.json").read_text())
        answers = [e["answer"] for e in splits[split]]
    results = [play_episode(solver, a, allowed) for a in answers]
    print(split, summarize(results))
