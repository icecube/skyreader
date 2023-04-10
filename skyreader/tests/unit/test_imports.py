"""Test objects are importable that should be importable."""


import skyreader


def test_skyreader_imports() -> None:
    """Test importing from 'skyreader'."""
    assert hasattr(skyreader, "scan_result")
    assert hasattr(skyreader, "SkyScanResult")
