"""Run: python -m pytest tests  (or python tests/test_env.py). No Jev calls."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from wordle_env.game import score_guess, WordleGame, InvalidGuess, play_episode
from wordle_env.data import make_splits


def test_feedback_basic():
    assert score_guess("crane", "crane") == "GGGGG"
    assert score_guess("arise", "rebus") == "BYBYY"   # proposal Figure 2, row 1


def test_feedback_duplicate_letters():
    assert score_guess("speed", "abide") == "BBYBY"   # only one E is yellow
    assert score_guess("eerie", "sweet") == "YYBBB"   # two Es in answer, three in guess
    assert score_guess("geese", "those") == "BBBGG"   # green E consumes the only extra copy
    assert score_guess("abbey", "kebab") == "YYGYB"


def test_game_rules():
    g = WordleGame("rebus", {"arise", "route", "rules", "rebus"})
    try:
        g.guess("zzzzz"); assert False
    except InvalidGuess:
        pass
    assert g.history == []                            # rejected guess not recorded
    assert "rebus" not in str(g.observation())        # hidden answer never observed
    for w in ["arise", "route", "rules", "rebus"]:
        g.guess(w)
    assert g.solved and g.turns_used() == 4


def test_unsolved_and_invalid_count_as_seven():
    allowed = {"rebus", "arise"}
    assert play_episode(lambda o: "arise", "rebus", allowed)["turns"] == 7
    r = play_episode(lambda o: "qqqqq", "rebus", allowed)
    assert r["status"] == "invalid_action" and r["turns"] == 7


def test_splits_disjoint_and_no_answer_in_id():
    words = [f"w{i:04d}"[:5] for i in range(1000)]
    s = make_splits(sorted(set(words)), seed=0)
    ids = [e["id"] for v in s.values() for e in v]
    assert len(ids) == len(set(ids))
    assert all(e["answer"] not in e["id"] for v in s.values() for e in v)
    assert sum(map(len, s.values())) == len(set(words))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
