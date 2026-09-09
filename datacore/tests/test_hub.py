"""
Test HubTable: in-memory pyarrow table wrapper + seeded shuffle (no network -- load_hub_dataset's
actual download path is exercised only by a host application against the real HF hub API).
Ported from nanochat/tests/test_tasks.py's HubDataset tests. Skips if pyarrow isn't installed
(the [parquet] extra).

python -m pytest datacore/tests/test_hub.py -v
"""
import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")

from datacore.hub import HubTable


def test_hub_table_rows():
    table = pa.table({"x": list(range(100)), "y": [str(i) for i in range(100)]})
    ds = HubTable(table)
    assert len(ds) == 100
    assert ds[7] == {"x": 7, "y": "7"}


def test_hub_table_shuffle_matches_numpy():
    # the shuffle must reproduce datasets.Dataset.shuffle(seed) exactly,
    # which is a np.random.default_rng(seed) permutation
    table = pa.table({"x": list(range(100))})
    ds = HubTable(table).shuffle(seed=42)
    perm = np.random.default_rng(42).permutation(100)
    assert [ds[i]["x"] for i in range(100)] == [int(p) for p in perm]
    # shuffling returns a view; the original order is untouched
    assert HubTable(table)[0] == {"x": 0}
