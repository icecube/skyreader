from constants import CATALOG_PATH, CATALOG_NAME

import skyreader


def test_skyreader_imports() -> None:
    """Test importing from 'skyreader'."""
    assert hasattr(skyreader, "EventMetadata")
    assert hasattr(skyreader, "SkyScanResult")
    assert hasattr(skyreader, "plot")

    expected_catalog_path = (
        "/cvmfs/icecube.opensciencegrid.org/users/azegarelli/realtime/"
        "catalogs/gll_psc_v35.fit"
    )
    assert CATALOG_PATH == expected_catalog_path
    assert CATALOG_NAME == "gll_psc_v35.fit"
