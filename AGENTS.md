# AGENTS.md

`datacore` is a standalone data subsystem — prepares a raw text/token corpus into a pretokenized,
packed, multipart on-disk dataset (`DataManager.prepare`) and reads it back as flexible-batch-size,
DDP-shardable, exactly-resumable batches (`DataManager.batches`). Read
[docs/architecture.md](docs/architecture.md) for the full contract before touching anything here.
For the family-wide pattern this repo follows (one entrypoint, zero host imports, the tag-pin
consumption contract) see [llmllab/AGENTS.md](../llmllab/AGENTS.md) and
[llmllab/docs/subsystem-conventions.md](../llmllab/docs/subsystem-conventions.md).

Its host application is [8kb/nanochat](https://github.com/8kb/nanochat), which pins this repo by
git tag (`pyproject.toml`'s `[tool.uv.sources]`) and consumes it entirely through `DataManager` —
see nanochat's own [docs/architecture.md](https://github.com/8kb/nanochat/blob/master/docs/architecture.md)
for that side, and `scripts/data_prep.py` there for the preparation entrypoint.

## Repo map

```
datacore/
├── manager.py         DataManager -- the one entrypoint (prepare / open / batches)
├── store.py            DatasetStore protocol + FileSystemDatasetStore; token_dtype()
├── packing.py           BestFitCropPacker / BestFitPadPacker
├── writer.py             write_split() -- turns tokenized text into packed .npy volumes
├── reader.py              Dataset/DatasetInfo/SplitIndex, open_dataset(), batches() -- the
│                          only module that imports torch, lazily, inside batches()
├── sources.py             TextSource/TokenSource protocols, ParquetDirectorySource
├── download.py             corpus download helper (stdlib urllib, no torch/requests)
├── tokenizer.py             Tokenizer/CharTokenizer protocol + reference implementation
│                            (token_byte_lengths() is an OPTIONAL fifth member)
└── tests/                   this repo's own test suite
```

## Invariants that will bite you

- **A prepared dataset's `sequence_len` and tokenizer fingerprint are fixed, and a caller's must
  match, or `batches()`/`open_dataset()` should be treated as raising, not warning.** This package
  itself doesn't own the tokenizer-identity check (a host provides its own tokenizer and compares
  its fingerprint against `dataset.info.tokenizer_fingerprint`), but the on-disk `sequence_len` is
  fixed at prepare time and every row is exactly that length — a caller reading at a different
  length is a bug, not a variant. Batch size, world size, rank, and split are the only things free
  at read time.
- **The dataloader state is an exact global sequence cursor, not an approximation.**
  `{"format": "datacore.v1", "cursor", "epoch", "num_sequences", "batch_size", "world_size"}` —
  `cursor` is the count of sequences consumed by all ranks so far, world-size-independent by
  construction: resuming `batches()` at a different `world_size` than the run that saved the state
  still produces a gap-free, duplicate-free continuation. A state dict with no `"format"` key
  predates this format entirely and has no faithful translation into a sequence cursor — a host
  reading one should refuse it explicitly rather than guess.
- **`reader.py` is the only module that imports torch, and only lazily, inside `batches()`.**
  `prepare()`/`writer.write_split()`/`packing.py`/`sources.py`/`download.py` must stay torch-free —
  forking a worker pool after torch has touched CUDA is a real hazard, and the write path has no
  use for a tensor anyway. A change that adds a torch import to any write-path module breaks this.
- **No ambient globals.** `rank`/`world_size`/`device` are always explicit parameters to
  `batches()`, never read from the environment — same rule `modelcore.runtime` applies to compute
  dtype (see [modelcore/AGENTS.md](../modelcore/AGENTS.md)), independently arrived at here.
- **The packing algorithms' parity proof lives in the host repo, not here.** nanochat's
  `tests/goldens/data_bestfit_{crop,pad}.json` (captured from its pre-datacore packing code by
  `dev/capture_data_goldens.py`, frozen) and `tests/test_data_packing_parity.py` cross-check
  `BestFitCropPacker`/`BestFitPadPacker` against that frozen reference — a change to `packing.py`
  needs that check run too, since this repo's own suite has no fixture proving byte-identical
  output to the pre-extraction algorithm.
- **The per-token byte-length table (`token_bytes.npy`, for a host's bits-per-byte eval) is
  entirely the tokenizer's data.** `prepare()` only calls the tokenizer's optional
  `token_byte_lengths()` and persists whatever comes back (see `docs/architecture.md`'s "Sources
  and the tokenizer interface") — datacore never derives byte lengths itself. A dataset prepared
  before this existed, or against a tokenizer without the method, has no artefact;
  `Dataset.token_bytes()`/`DataManager.token_bytes()` raise rather than guess. There is no
  backfill path — re-prepare the dataset.

## Testing

```bash
python -m pytest datacore/tests -v
```

No GPU, no real tokenizer required — `CharTokenizer` and `torch`'s CPU path cover the whole suite;
`pyarrow`-dependent tests skip automatically if it isn't installed. `datacore/tests/test_standalone.py`
mechanically checks that nothing under `datacore/` imports a host application (or `modelcore`, a
sibling standalone component) — see
[docs/architecture.md#verifying-a-change-is-behavior-preserving](docs/architecture.md#verifying-a-change-is-behavior-preserving)
for the from-scratch standalone-copy recipe.

A change here that a host application depends on needs that host's own suite run against it too —
for nanochat, `tests/test_data_packing_parity.py` (and the base-train/SFT smoke path) after an
editable install (`uv pip install -e ../datacore` from nanochat's venv) — this repo's own tests
proving *it* still works is necessary but not sufficient proof the host is unaffected.

## Style

See [llmllab/AGENTS.md](../llmllab/AGENTS.md#style).
