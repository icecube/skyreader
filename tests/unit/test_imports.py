"""Test objects are importable that should be importable."""


import skyreader


def test_skyreader_imports() -> None:
    """Test importing from 'skyreader'."""
    assert hasattr(skyreader, "EventMetadata")
    assert hasattr(skyreader, "SkyScanResult")
    assert hasattr(skyreader, "plot")
