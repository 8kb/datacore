"""Shared fixtures for datacore's test suite."""
import pytest

from datacore import CharTokenizer

from datacore.tests.helpers import CHARS


@pytest.fixture
def char_tokenizer():
    return CharTokenizer(CHARS)
