"""
Test ExampleSet/ExampleMixture/ExampleSequence: slicing views and combinators (in-memory, no
network). Ported from nanochat/tests/test_tasks.py's container-logic tests (its HubDataset/
render_mc tests moved to test_hub.py / stayed in benchcore respectively).

python -m pytest datacore/tests/test_records.py -v
"""
from datacore.records import ExampleMixture, ExampleSequence, ExampleSet


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


def test_sequence_is_concatenation_in_order():
    sequence = ExampleSequence([ToyExampleSet(n=3, tag="a"), ToyExampleSet(n=2, tag="b")])
    assert len(sequence) == 5
    tags = [sequence[i]["tag"] for i in range(5)]
    assert tags == ["a", "a", "a", "b", "b"]
    idxs = [sequence[i]["i"] for i in range(5)]
    assert idxs == [0, 1, 2, 0, 1]
