import io
import pytest
import pandas as pd
import numpy as np
from modules.data_loader import load_dataset, get_file_info

class MockUploadedFile:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content
        self._buffer = io.BytesIO(content)

    def read(self, *args):
        return self._buffer.read(*args)

    def seek(self, pos, whence=0):
        return self._buffer.seek(pos, whence)

    def tell(self):
        return self._buffer.tell()

    @property
    def size(self):
        return len(self._content)


def test_load_csv_success():
    csv_bytes = b"feature1,feature2,target\n1.0,2.0,A\n3.0,4.0,B\n5.0,6.0,A\n"
    mock_file = MockUploadedFile("test_data.csv", csv_bytes)

    df, file_info, error = load_dataset(mock_file)
    assert error is None
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 3)
    assert list(df.columns) == ["feature1", "feature2", "target"]
    assert file_info["name"] == "test_data.csv"


def test_load_json_success():
    json_bytes = b'[{"a": 1, "b": 2}, {"a": 3, "b": 4}]'
    mock_file = MockUploadedFile("data.json", json_bytes)

    df, file_info, error = load_dataset(mock_file)
    assert error is None
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)


def test_load_unsupported_format():
    bin_bytes = b"\x00\x01\x02\x03\x04"
    mock_file = MockUploadedFile("archive.bin", bin_bytes)

    df, file_info, error = load_dataset(mock_file)
    assert df is None
    assert error is not None
    assert "Unsupported" in error or "unsupported" in error


def test_get_file_info():
    mock_file = MockUploadedFile("sample.csv", b"a,b\n1,2\n")
    info = get_file_info(mock_file)
    assert info["name"] == "sample.csv"
    assert info["extension"] == ".csv"
    assert info["size_bytes"] == len(b"a,b\n1,2\n")

