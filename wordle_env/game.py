"""Wordle game core. No JevHarness dependency, so the baselines can reuse it."""
from collections import Counter
import time

GREEN, YELLOW, GRAY = "G", "Y", "B"
MAX_TURNS = 6
FAIL_TURNS = 7  # proposal: an unsolved game counts as 7 turns


def score_guess(guess, answer):
    """Return feedback such as 'GYBBG'.

    Two passes, so duplicate letters follow the official rules: greens are
    assigned first, then yellows only while unmatched copies remain.
    """
    result = [GRAY] * 5
    unmatched = Counter()
    for i, (g, a) in enumerate(zip(guess, answer)):
        if g == a:
            result[i] = GREEN
        else:
            unmatched[a] += 1
    for i, g in enumerate(guess):
        if result[i] != GREEN and unmatched[g] > 0:
            result[i] = YELLOW
            unmatched[g] -= 1
    return "".join(result)


class InvalidGuess(ValueError):
    """The policy proposed something that is not an allowed 5-letter word."""


class WordleGame:
    def __init__(self, answer, allowed, max_turns=MAX_TURNS):
        if answer not in allowed:
            raise ValueError("answer must be in the allowed-guess list")
        self._answer = answer  # private: never placed in an observation
        self.allowed = allowed
        self.max_turns = max_turns
        self.history = []
        self.solved = False

    @property
    def done(self):
        return self.solved or len(self.history) >= self.max_turns

    def guess(self, word):
        if self.done:
            raise RuntimeError("game is already over")
        if not isinstance(word, str):
            raise InvalidGuess(f"guess must be a string, got {type(word).__name__}")
        word = word.strip().lower()
        if word not in self.allowed:
            raise InvalidGuess(f"{word!r} is not an allowed guess")
        feedback = score_guess(word, self._answer)
        self.history.append({"guess": word, "feedback": feedback})
        self.solved = feedback == GREEN * 5
        return feedback

    def turns_used(self):
        return len(self.history) if self.solved else FAIL_TURNS

    def observation(self):
        """Player-visible state only."""
        return {
            "turn": len(self.history) + 1,
            "max_turns": self.max_turns,
            "history": [dict(h) for h in self.history],
        }


def play_episode(policy, answer, allowed):
    """Run any policy(obs) -> word. Shared by the LLM and rule-based baselines.

    An invalid guess ends the game as unsolved (7 turns); the same rule is
    applied to the Jev harness in adapter.py so all methods are comparable.
    """
    game = WordleGame(answer, allowed)
    decision_ms, status, error = [], "completed", None
    while not game.done:
        t0 = time.perf_counter()
        word = policy(game.observation())
        decision_ms.append((time.perf_counter() - t0) * 1000)
        try:
            game.guess(word)
        except InvalidGuess as exc:
            status, error = "invalid_action", str(exc)
            break
    turns = game.turns_used() if status == "completed" else FAIL_TURNS
    return {"answer": answer, "status": status, "error": error,
            "solved": game.solved and status == "completed", "turns": turns,
            "history": game.history, "decision_ms": decision_ms}
