import pytest

from datacore import CharTokenizer
from datacore.packing import EncodedDoc
from datacore.records import ExampleSet
from datacore.sources import ExampleTokenSource, named_document_batches

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

CHARS = " abcdefghijklmnopqrstuvwxyz.,!?'\n0123456789"


class ListTextSource:
    def __init__(self, batches):
        self._batches = batches

    def text_batches(self):
        return iter(self._batches)


class ListTokenSource:
    def __init__(self, batches):
        self._batches = batches

    def token_batches(self):
        return iter(self._batches)


def test_text_source_gets_bos_prepended():
    tok = CharTokenizer(CHARS)
    source = ListTextSource([("f", ["hi", "bye"])])
    out = list(named_document_batches(source, tok))
    assert len(out) == 1
    name, docs = out[0]
    docs = list(docs)
    assert name == "f"
    assert docs[0].ids[0] == tok.get_bos_token_id()
    assert docs[1].ids[0] == tok.get_bos_token_id()


def test_token_source_passes_through_unchanged():
    tok = CharTokenizer(CHARS)
    doc = EncodedDoc(ids=[1, 2, 3], mask=[0, 1, 1])
    source = ListTokenSource([("f", [doc])])
    out = list(named_document_batches(source, tok))
    name, docs = out[0]
    docs = list(docs)
    assert docs[0] is doc


def test_unrecognized_source_raises():
    tok = CharTokenizer(CHARS)
    with pytest.raises(TypeError):
        list(named_document_batches(object(), tok))


class ToyRecordSet(ExampleSet):
    """Each example is just its own index -- render turns it into a trivial (ids, mask) pair."""
    def __init__(self, n, **kwargs):
        super().__init__(**kwargs)
        self.n = n

    def num_examples(self):
        return self.n

    def get_example(self, index):
        return index


def _render(record):
    return [record, record + 1], [0, 1]


def test_example_token_source_renders_every_record():
    source = ExampleTokenSource(ToyRecordSet(3), _render, "train")
    batches = list(source.token_batches())
    assert len(batches) == 1  # one chunk, since chunk_size (2000) > 3
    name, docs = batches[0]
    assert name == "train[0:3]"
    docs = list(docs)
    assert [d.ids for d in docs] == [[0, 1], [1, 2], [2, 3]]
    assert [d.mask for d in docs] == [[0, 1], [0, 1], [0, 1]]


def test_example_token_source_respects_chunk_boundaries():
    source = ExampleTokenSource(ToyRecordSet(5), _render, "train", chunk_size=2)
    batches = list(source.token_batches())
    assert [name for name, _ in batches] == ["train[0:2]", "train[2:4]", "train[4:5]"]
    total_docs = sum(len(list(docs)) for _, docs in batches)
    assert total_docs == 5


def test_example_token_source_satisfies_token_source_protocol():
    tok = CharTokenizer(" abcdefghijklmnopqrstuvwxyz.,!?'\n0123456789")
    source = ExampleTokenSource(ToyRecordSet(2), _render, "train")
    out = list(named_document_batches(source, tok))
    assert len(out) == 1
    name, docs = out[0]
    assert name == "train[0:2]"
    assert [d.ids for d in docs] == [[0, 1], [1, 2]]


def test_parquet_directory_source(tmp_path):
    from datacore.sources import ParquetDirectorySource
    path = tmp_path / "shard_00000.parquet"
    table = pa.Table.from_pydict({"text": ["doc one", "doc two", "doc three"]})
    pq.write_table(table, str(path), row_group_size=2)
    source = ParquetDirectorySource(paths=[str(path)])
    batches = list(source.text_batches())
    assert len(batches) == 1
    name, texts = batches[0]
    assert name == str(path)
    assert list(texts) == ["doc one", "doc two", "doc three"]
