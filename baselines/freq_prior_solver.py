"""Setting-B baseline: entropy solver with a word-frequency prior (code only).

The answer list is unknown, so all 12972 legal words are candidates. Each
candidate gets prior weight  w = sigmoid((zipf - mid) / scale), where zipf is
the English word frequency from the `wordfreq` package (0 if unknown).

Guess choice minimises an estimate of the expected number of turns left:

    T(g) = 1 + sum_{b != GGGGG} P(b) * (1 + c * H_b)

P(b) is the prior mass of feedback bucket b, H_b the entropy (bits) of the
normalised prior inside b, and "1 + c*H" a crude model of how many more
guesses a bucket of entropy H needs. Guessing a likely candidate lowers T
because its GGGGG bucket costs nothing, so the solver trades "go for the win"
against "split the candidates", which plain entropy cannot.

mid, scale and c are tuned on the train split only (see `tune`).
"""
import itertools
import json
import sys
from statistics import mean

import numpy as np
from wordfreq import zipf_frequency

from baselines.entropy_solver import CHUNK, EntropySolver, feedback_code
from baselines.optimal_tree import ROOT, summarize
from wordle_env.data import load_words
from wordle_env.game import play_episode

WIN = feedback_code("GGGGG")


class FreqPriorSolver(EntropySolver):
    def __init__(self, mid=3.0, scale=0.5, c=0.5, opener=None):
        super().__init__(pool="allowed", opener="(unset)")
        self.zipf = np.array([zipf_frequency(w, "en") for w in self.answers])
        self.prior = 1 / (1 + np.exp(-(self.zipf - mid) / scale))
        self.c = c
        self.opener = opener or self.best_guess(np.arange(len(self.answers)))

    def best_guess(self, cand):
        w = self.prior[cand]
        if len(cand) == 1:
            return self.answers[cand[0]]
        wlogw = w * np.log2(w)
        ng = len(self.guesses)
        cost = np.empty(ng)
        for lo in range(0, ng, CHUNK):
            sub = self.M[lo:lo + CHUNK][:, cand].astype(np.int64)
            k = len(sub)
            flat = (np.arange(k)[:, None] * 243 + sub).ravel()
            W = np.bincount(flat, weights=np.tile(w, k), minlength=k * 243).reshape(k, 243)
            S = np.bincount(flat, weights=np.tile(wlogw, k), minlength=k * 243).reshape(k, 243)
            with np.errstate(divide="ignore", invalid="ignore"):
                H = np.where(W > 0, np.log2(W) - S / W, 0.0)   # entropy inside each bucket
            P = W / w.sum()
            P[:, WIN] = 0.0                                     # guessed it: no turns left
            cost[lo:lo + k] = 1 + (P * (1 + self.c * H)).sum(axis=1)
        return self.guesses[int(np.argmin(cost))]


def run(solver, answers, allowed):
    return summarize([play_episode(solver, a, allowed) for a in answers])


def tune(train, allowed, grid):
    best = None
    for mid, scale, c in grid:
        s = run(FreqPriorSolver(mid, scale, c), train, allowed)
        print(f"mid={mid} scale={scale} c={c} -> {s['mean_turns']:.4f} (solved {s['solved']})", flush=True)
        if best is None or s["mean_turns"] < best[0]:
            best = (s["mean_turns"], (mid, scale, c))
    return best


if __name__ == "__main__":
    answers, allowed = load_words()
    splits = {k: [e["answer"] for e in v]
              for k, v in json.loads((ROOT / "data" / "splits.json").read_text()).items()}
    if sys.argv[1:2] == ["tune"]:
        grid = itertools.product([2.0, 2.5, 3.0], [0.25, 0.5, 1.0], [0.3, 0.5, 0.7])
        print("best on train:", tune(splits["train"], allowed, grid))
    else:
        mid, scale, c = (float(x) for x in sys.argv[1:4])
        solver = FreqPriorSolver(mid, scale, c)
        print("params", (mid, scale, c), "opener:", solver.opener)
        for name in ["train", "select", "test"]:
            print(name, run(solver, splits[name], allowed))
        print("all", run(solver, answers, allowed))
