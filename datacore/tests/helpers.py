"""Shared test helpers: the CharTokenizer alphabet, a manifest builder for hand-written datasets, and
a raw row reader."""
CHARS = " abcdefghijklmnopqrstuvwxyz.,!?'\n0123456789"


def write_manifest(store, packer, totals, *, sequence_len, vocab_size, bos_token_id, fingerprint="t"):
    """Writes the manifest DataManager.prepare would, for a dataset a test wrote with write_split."""
    params = {}
    if hasattr(packer, "padding_id"):
        params["padding_id"] = packer.padding_id
    manifest = {
        "format": "datacore.v1", "sequence_len": sequence_len, "dtype": "uint16",
        "has_mask": bool(packer.emits_mask), "vocab_size": vocab_size, "bos_token_id": bos_token_id,
        "tokenizer_fingerprint": fingerprint, "packer": {"name": packer.name, "params": params},
        "splits": {"train": {
            "volumes": [{"file": v.file, "rows": v.rows, "source": v.source,
                         **({"mask_file": v.mask_file} if v.mask_file else {})} for v in totals.volumes],
            "num_sequences": totals.num_sequences, "num_tokens": totals.num_tokens,
            "num_documents": totals.num_documents, "num_documents_dropped": totals.num_documents_dropped,
            "num_tokens_encoded": totals.num_tokens_encoded, "num_tokens_dropped": totals.num_tokens_dropped,
        }},
    }
    store.write_manifest(manifest)
    return manifest


def read_rows(dataset, split, start, count):
    """Raw (tokens, mask) for rows [start, start+count) of a split -- mask is None without one."""
    return dataset._split_indices[split].read_contiguous(start, count)
