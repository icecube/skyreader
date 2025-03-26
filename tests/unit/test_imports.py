import skyreader
from skyreader.constants import CATALOG_PATH


def test_skyreader_imports() -> None:
    """Test importing from 'skyreader'."""
    assert hasattr(skyreader, "EventMetadata")
    assert hasattr(skyreader, "SkyScanResult")
    assert hasattr(skyreader, "plot")


def fermi_catalog_import() -> None:
    expected_catalog_path = (
        "/cvmfs/icecube.opensciencegrid.org/users/azegarelli/realtime/"
        "catalogs/gll_psc_v35.fit"
    )

    assert CATALOG_PATH == expected_catalog_path
