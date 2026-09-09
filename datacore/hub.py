"""
HubTable + load_hub_dataset: read a HuggingFace Hub dataset's auto-generated parquet export into
an in-memory table with lazy row access and a seeded shuffle. Ported from
nanochat/tasks/common.py's HubDataset/load_hub_dataset -- a mechanism (list shards via the hub
API, download once, read with pyarrow) parameterized entirely by identity (repo_id/subset/split
and, now, an explicit cache_dir), the same shape as datacore.download.download_shards.

Which repo_id/subset/split to load, and how many epochs to draw from it, remains a host
application's call (e.g. benchcore's tasks/*.py or nanochat's own SFT mixture) -- datacore only
owns the download-once-and-read-back mechanism.
"""
import json
import os
import urllib.request

import numpy as np
from filelock import FileLock

# pyarrow is only imported inside load_hub_dataset, lazily -- like sources.ParquetDirectorySource,
# this keeps the [parquet] extra optional for callers who never touch hub datasets.


class HubTable:
    """
    Minimal stand-in for a HuggingFace datasets Dataset: wraps a pyarrow Table and offers lazy
    row access and a seeded shuffle.
    """

    def __init__(self, table, permutation=None):
        self.table = table
        self.permutation = permutation

    def __len__(self):
        return self.table.num_rows

    def shuffle(self, seed):
        # matches datasets.Dataset.shuffle(seed=seed) exactly, row order comes out identical
        permutation = np.random.default_rng(seed).permutation(len(self))
        return HubTable(self.table, permutation)

    def __getitem__(self, index):
        physical_index = index if self.permutation is None else int(self.permutation[index])
        return {column: self.table[column][physical_index].as_py() for column in self.table.column_names}


def load_hub_dataset(repo_id, subset="default", split="train", *, cache_dir):
    """
    Minimal stand-in for HuggingFace datasets.load_dataset(repo_id, subset, split=split).
    Every dataset on the hub has an auto-generated parquet export. We list the parquet
    shards via the hub API, download them (once) into cache_dir, and read them with pyarrow.
    Under multi-process launches, only one process downloads (via FileLock); the others block
    then skip the download because they recheck the manifest.

    cache_dir: where to cache downloaded shards -- an explicit parameter, not read from an
    ambient global (a host application's own base directory, e.g. nanochat's get_base_dir()).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    slug = repo_id.replace("/", "--")
    # "task_data" (not e.g. "hub_data") matches nanochat's original tasks/common.py::load_hub_dataset
    # exactly, so a pre-existing local cache from before this moved here is reused, not re-downloaded.
    shards_dir = os.path.join(cache_dir, "task_data", slug, subset, split)
    # the manifest is written last, so its existence means the download completed
    manifest_path = os.path.join(shards_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        os.makedirs(shards_dir, exist_ok=True)
        with FileLock(manifest_path + ".lock"):
            # only a single process acquires the lock and downloads, the others block
            # here and then skip the download because they recheck the manifest
            if not os.path.exists(manifest_path):
                listing_url = f"https://huggingface.co/api/datasets/{repo_id}/parquet/{subset}/{split}"
                with urllib.request.urlopen(listing_url) as response:
                    shard_urls = json.loads(response.read())
                filenames = []
                for shard_index, shard_url in enumerate(shard_urls):
                    filename = f"{shard_index:05d}.parquet"
                    print(f"Downloading {shard_url} ...")
                    with urllib.request.urlopen(shard_url) as response:
                        content = response.read()
                    with open(os.path.join(shards_dir, filename), "wb") as f:
                        f.write(content)
                    filenames.append(filename)
                with open(manifest_path, "w") as f:
                    json.dump(filenames, f)
    with open(manifest_path, "r") as f:
        filenames = json.load(f)
    shard_paths = [os.path.join(shards_dir, filename) for filename in filenames]
    tables = [pq.read_table(path) for path in shard_paths]
    table = pa.concat_tables(tables)
    return HubTable(table)
