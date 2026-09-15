"""
DatasetMismatch: the one typed error DataManager.open raises when a caller-supplied expectation
(sequence_len, tokenizer fingerprint) doesn't match what's actually on disk. datacore itself never
decides *to* compare these -- see DataManager.open's docstring and datacore/AGENTS.md -- a caller
that doesn't pass expect_sequence_len/expect_fingerprint never sees this at all. Typed (rather than
a plain ValueError) so a caller can catch it specifically and write its own remediation text
without string-matching a message.
"""
from dataclasses import dataclass


@dataclass
class DatasetMismatch(Exception):
    reason: str      # "sequence_len" | "tokenizer_fingerprint"
    expected: object
    actual: object

    def __str__(self):
        return f"{self.reason} mismatch: expected {self.expected!r}, dataset has {self.actual!r}"
