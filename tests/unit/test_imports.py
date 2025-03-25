from skyreader.plots.constants import CATALOG_PATH

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
