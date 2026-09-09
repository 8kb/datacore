"""
ExampleSet: a lightweight, sliceable view over an indexable collection of records, plus
ExampleMixture/ExampleSequence combinators. Ported verbatim (container logic only) from
nanochat/tasks/common.py's Task/TaskMixture/TaskSequence -- the eval-specific half (eval_type,
evaluate(), reward()) stayed in the host's eval subsystem (benchcore), since datacore has no
business knowing what "correct" means for a record.

Not an ABC -- duck typing is enough, and datacore has no business enforcing what a caller's record
collection subclasses from. A record is whatever get_example() returns; datacore never looks
inside one.
"""
import random


class ExampleSet:
    """
    Base class for an indexable collection of records. Allows for lightweight logical slicing
    (start/stop/step) over an underlying collection without copying it.
    """

    def __init__(self, start=0, stop=None, step=1):
        assert start >= 0, f"Start must be non-negative, got {start}"
        assert stop is None or stop >= start, f"Stop should be greater than or equal to start, got {stop} and {start}"
        assert step >= 1, f"Step must be strictly positive, got {step}"
        self.start = start
        self.stop = stop # could be None here
        self.step = step

    def num_examples(self):
        raise NotImplementedError

    def get_example(self, index):
        raise NotImplementedError

    def __len__(self):
        start = self.start
        stop = self.num_examples() if self.stop is None else self.stop
        step = self.step
        span = stop - start
        num = (span + step - 1) // step # ceil_div(span, step)
        assert num >= 0, f"Negative number of examples???: {num}" # prevent footguns
        return num

    def __getitem__(self, index: int):
        assert isinstance(index, int), f"Index must be an integer, got {type(index)}"
        physical_index = self.start + index * self.step
        return self.get_example(physical_index)


class ExampleMixture(ExampleSet):
    """
    Mixes multiple ExampleSets into one, deterministically shuffled so the sets interleave
    throughout instead of appearing as contiguous blocks.
    Fun trick: if you wish to oversample any set, just pass it in multiple times in the list.
    """

    def __init__(self, sets, **kwargs):
        super().__init__(**kwargs)
        self.sets = sets
        self.lengths = [len(s) for s in self.sets]
        self.num_examples_total = sum(self.lengths)
        # Build list of all (set_idx, local_idx) pairs
        self.index_map = []
        for set_idx, set_length in enumerate(self.lengths):
            for local_idx in range(set_length):
                self.index_map.append((set_idx, local_idx))
        # Deterministically shuffle so the sets are mixed throughout, not concatenated
        rng = random.Random(42)
        rng.shuffle(self.index_map)
        # Note: this is not the most elegant or best solution, but it's ok for now

    def num_examples(self):
        return self.num_examples_total

    def get_example(self, index):
        assert 0 <= index < self.num_examples_total, f"Index {index} out of range for mixture with {self.num_examples_total} examples"
        set_idx, local_idx = self.index_map[index]
        return self.sets[set_idx][local_idx]


class ExampleSequence(ExampleSet):
    """
    Sequentially concatenates a list of ExampleSets. Useful for curricula that require a fixed
    order rather than a mixture's interleave.
    """

    def __init__(self, sets, **kwargs):
        super().__init__(**kwargs)
        self.sets = sets
        self.lengths = [len(s) for s in self.sets]
        self.num_examples_total = sum(self.lengths)

    def num_examples(self):
        return self.num_examples_total

    def get_example(self, index):
        assert 0 <= index < self.num_examples_total, f"Index {index} out of range for sequence with {self.num_examples_total} examples"
        for set_idx, set_length in enumerate(self.lengths):
            if index < set_length:
                return self.sets[set_idx][index]
            index -= set_length
