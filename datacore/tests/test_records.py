"""
Test ExampleSet/ExampleMixture: slicing views and combinators (in-memory, no network). Ported from
our nanochat fork's tests/test_tasks.py.

python -m pytest datacore/tests/test_records.py -v
"""
from datacore.records import ExampleMixture, ExampleSet


class ToyExampleSet(ExampleSet):
    """A trivial set: example i is just {'i': i, 'tag': tag}."""

    def __init__(self, n=10, tag="a", **kwargs):
        super().__init__(**kwargs)
        self.n = n
        self.tag = tag

    def num_examples(self):
        return self.n

    def get_example(self, index):
        return {"i": index, "tag": self.tag}


def test_example_set_full():
    s = ToyExampleSet(n=10)
    assert len(s) == 10
    assert s[0] == {"i": 0, "tag": "a"}
    assert s[9] == {"i": 9, "tag": "a"}


def test_example_set_slicing():
    # a view of [5, 10) has 5 examples and maps logical to physical indices
    s = ToyExampleSet(n=10, start=5, stop=10)
    assert len(s) == 5
    assert s[0]["i"] == 5
    # step slicing uses ceil division for the length
    s = ToyExampleSet(n=10, start=0, stop=10, step=3) # 0, 3, 6, 9
    assert len(s) == 4
    assert [s[i]["i"] for i in range(4)] == [0, 3, 6, 9]


def test_example_set_stop_clamps_to_true_length():
    """An over-large explicit stop is clamped, not taken at face value -- this is what lets a
    caller pass stop=<some cap> without separately computing min(cap, len) first (no wrapper class needed)."""
    s = ToyExampleSet(n=10, stop=1000)
    assert len(s) == 10
    assert [s[i]["i"] for i in range(10)] == list(range(10))
    # stop below the true length still works as a normal cap
    s2 = ToyExampleSet(n=10, stop=4)
    assert len(s2) == 4


def test_mixture_stop_caps_and_clamps():
    mixture = ExampleMixture([ToyExampleSet(n=3, tag="a"), ToyExampleSet(n=5, tag="b")], stop=1000)
    assert len(mixture) == 8  # clamped to the true total, not 1000
    capped = ExampleMixture([ToyExampleSet(n=3, tag="a"), ToyExampleSet(n=5, tag="b")], stop=3)
    assert len(capped) == 3
    assert [capped[i] for i in range(3)] == [mixture[i] for i in range(3)]


def test_mixture_covers_all_examples_deterministically():
    mixture = ExampleMixture([ToyExampleSet(n=3, tag="a"), ToyExampleSet(n=5, tag="b")])
    assert len(mixture) == 8
    examples = [mixture[i] for i in range(8)]
    # every example appears exactly once
    keys = sorted((ex["tag"], ex["i"]) for ex in examples)
    assert keys == [("a", 0), ("a", 1), ("a", 2), ("b", 0), ("b", 1), ("b", 2), ("b", 3), ("b", 4)]
    # the shuffle is deterministic: a second instance yields the same order
    mixture2 = ExampleMixture([ToyExampleSet(n=3, tag="a"), ToyExampleSet(n=5, tag="b")])
    assert examples == [mixture2[i] for i in range(8)]
    # and the sets are actually interleaved, not concatenated
    assert [ex["tag"] for ex in examples] != ["a"] * 3 + ["b"] * 5


def test_mixture_oversampling():
    # passing a set twice doubles its examples
    mixture = ExampleMixture([ToyExampleSet(n=3), ToyExampleSet(n=3)])
    assert len(mixture) == 6


def test_mixture_seed_changes_the_interleave_and_the_default_is_stable():
    sets = [ToyExampleSet(n=5, tag="a"), ToyExampleSet(n=5, tag="b")]
    order = lambda m: [(m[i]["tag"], m[i]["i"]) for i in range(len(m))]
    assert order(ExampleMixture(sets)) == order(ExampleMixture(sets, seed=42))
    assert order(ExampleMixture(sets, seed=1)) != order(ExampleMixture(sets, seed=42))
