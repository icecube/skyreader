"""Test `is_close()`."""


from pathlib import Path
from typing import Any

import numpy as np
import pytest
from skyreader import SkyScanResult
from skyreader.result import PyDictResult


COLUMNS_V0 = ["index", "llh", "E_in", "E_tot"]
COLUMNS_V1 = ["index", "llh", "E_in", "E_tot", "X", "Y", "Z", "T"]


@pytest.fixture
def json_diff(request: Any) -> Path:
    """Use the tests name to create a json filename."""
    return Path(request.node.name + ".json")


def test_000(json_diff: Path) -> None:
    """Compare same instances."""
    # rtol_per_field = dict(llh=0.5, E_in=0.5, E_tot=0.5)

    alpha_pydict: PyDictResult = {
        "nside-8": {
            "columns": COLUMNS_V0,
            "metadata": {"nside": 8},
            "data": [
                [0, 496.5, 4643.5, 4736.5],
            ],
        },
    }
    alpha = SkyScanResult.deserialize(alpha_pydict)

    assert alpha.is_close(
        alpha,
        equal_nan=True,
        dump_json_diff=json_diff,
    )
    assert alpha.is_close(
        alpha,
        equal_nan=True,
        dump_json_diff=json_diff,
        rtol_per_field={}
    )


def test_001(json_diff: Path) -> None:
    """Compare same instances."""

    alpha_pydict: PyDictResult = {
        "nside-8": {
            "columns": COLUMNS_V1,
            "metadata": {"nside": 8,
                         "version": 1,
                         "run_id": 0,
                         "event_id": 0,
                         "mjd": 1.,
                         "event_type": ""},
            "data": [
                [0, 496.5, 4643.5, 4736.5, 1., 2., 3., 4.],
            ],
        },
    }
    alpha = SkyScanResult.deserialize(alpha_pydict)

    assert alpha.is_close(
        alpha,
        equal_nan=True,
        dump_json_diff=json_diff,
    )


def test_002(json_diff: Path) -> None:
    """Compare v0 with v1 data format."""
    alpha_pydict: PyDictResult = {
        "nside-8": {
            "columns": COLUMNS_V0,
            "metadata": {"nside": 8},
            "data": [
                [0, 496.5, 4643.5, 4736.5],
            ],
        },
    }
    alpha = SkyScanResult.deserialize(alpha_pydict)

    beta_pydict: PyDictResult = {
        "nside-8": {
            "columns": COLUMNS_V1,
            "metadata": {"nside": 8,
                         "version": 1,
                         "run_id": 0,
                         "event_id": 0,
                         "mjd": 1.,
                         "event_type": ""},
            "data": [
                [0, 496.5, 4643.5, 4736.5, 0., 0., 0., 0.],
            ],
        },
    }
    beta = SkyScanResult.deserialize(beta_pydict)

    assert alpha.is_close(
        beta,
        equal_nan=True,
        dump_json_diff=json_diff,
    )
    assert beta.is_close(
        alpha,
        equal_nan=True,
        dump_json_diff=json_diff,
    )


def test_010(json_diff: Path) -> None:
    """Compare two simple instances."""
    rtol_per_field = dict(llh=0.5, E_in=0.5, E_tot=0.5)

    alpha_pydict: PyDictResult = {
        "nside-8": {
            "columns": COLUMNS_V0,
            "metadata": {"nside": 8},
            "data": [
                [0, 496.5, 4643.5, 4736.5],
            ],
        },
    }
    alpha = SkyScanResult.deserialize(alpha_pydict)

    beta_pydict: PyDictResult = {
        "nside-8": {
            "columns": COLUMNS_V0,
            "metadata": {"nside": 8},
            "data": [
                [
                    r[0],
                    r[1] * (1 + rtol_per_field["llh"]),
                    r[2] * (1 + rtol_per_field["E_in"]),
                    r[3] * (1 + rtol_per_field["E_tot"]),
                ]
                for r in alpha_pydict["nside-8"]["data"]
            ],
        },
    }
    beta = SkyScanResult.deserialize(beta_pydict)

    assert alpha.is_close(
        beta,
        equal_nan=True,
        dump_json_diff=json_diff,
        rtol_per_field=rtol_per_field,
    )
    assert beta.is_close(
        alpha,
        equal_nan=True,
        dump_json_diff=json_diff,
        rtol_per_field=rtol_per_field,
    )


@pytest.mark.parametrize("fail_index", [0, 1])
@pytest.mark.parametrize("fail_field", ["llh", "E_in", "E_tot"])
def test_011__fail(fail_index: int, fail_field: str, json_diff: Path) -> None:
    """Compare two simple instances."""
    rtol_per_field = dict(llh=0.5, E_in=0.5, E_tot=0.5)

    def scale(field: str, index: int, fail_scale: float) -> float:
        if index != fail_index:
            return 1.0
        if field == fail_field:
            return fail_scale
        return 1.0

    alpha_pydict: PyDictResult = {
        "nside-8": {
            "columns": COLUMNS_V0,
            "metadata": {"nside": 8},
            "data": [
                [0, 496.5, 4643.5, 4736.5],
                [1, 586.5, 6845.5, 7546.5],
            ],
        },
    }
    alpha = SkyScanResult.deserialize(alpha_pydict)

    # figure how to scale values to fail for alpha vs. BIGGER (not symmetrical)

    fail_scale = (1 / rtol_per_field[fail_field]) * 1.1  # > 1/rtol should fail
    bigger_pydict: PyDictResult = {
        "nside-8": {
            "columns": COLUMNS_V0,
            "metadata": {"nside": 8},
            "data": [
                [
                    r[0],
                    r[1] * (1 + scale("llh", i, fail_scale) * rtol_per_field["llh"]),
                    r[2] * (1 + scale("E_in", i, fail_scale) * rtol_per_field["E_in"]),
                    r[3]
                    * (1 + scale("E_tot", i, fail_scale) * rtol_per_field["E_tot"]),
                ]
                for i, r in enumerate(alpha_pydict["nside-8"]["data"])
            ],
        },
    }
    bigger = SkyScanResult.deserialize(bigger_pydict)
    assert not alpha.is_close(
        bigger,
        equal_nan=True,
        dump_json_diff=json_diff,
        rtol_per_field=rtol_per_field,
    )

    # figure how to scale values to fail for BIGGER vs. alpha (not symmetrical)

    fail_scale = 2.0  # >1 should fail
    bigger_pydict = {
        "nside-8": {
            "columns": COLUMNS_V0,
            "metadata": {"nside": 8},
            "data": [
                [
                    r[0],
                    r[1] * (1 + scale("llh", i, fail_scale) * rtol_per_field["llh"]),
                    r[2] * (1 + scale("E_in", i, fail_scale) * rtol_per_field["E_in"]),
                    r[3]
                    * (1 + scale("E_tot", i, fail_scale) * rtol_per_field["E_tot"]),
                ]
                for i, r in enumerate(alpha_pydict["nside-8"]["data"])
            ],
        },
    }
    bigger = SkyScanResult.deserialize(bigger_pydict)
    assert not bigger.is_close(
        alpha,
        equal_nan=True,
        dump_json_diff=json_diff,
        rtol_per_field=rtol_per_field,
    )


def test_020(json_diff: Path) -> None:
    """Compare two multi-nside instances."""
    rtol_per_field = dict(llh=0.5, E_in=0.5, E_tot=0.5)

    alpha_pydict: PyDictResult = {
        "nside-8": {
            "columns": COLUMNS_V0,
            "metadata": {"nside": 8},
            "data": [
                [0, 496.5, 4643.5, 4736.5],
                [1, 586.5, 6845.5, 7546.5],
            ],
        },
        "nside-64": {
            "columns": COLUMNS_V0,
            "metadata": {"nside": 64},
            "data": [
                [0, 355.5, 4585.5, 7842.5],
                [1, 454.5, 8421.5, 5152.5],
                [2, 321.5, 7456.5, 2485.5],
            ],
        },
    }
    alpha = SkyScanResult.deserialize(alpha_pydict)

    beta_pydict: PyDictResult = {
        "nside-8": {
            "columns": COLUMNS_V0,
            "metadata": {"nside": 8},
            "data": [
                [
                    r[0],
                    r[1] * (1 + rtol_per_field["llh"]),
                    r[2] * (1 + rtol_per_field["E_in"]),
                    r[3] * (1 + rtol_per_field["E_tot"]),
                ]
                for r in alpha_pydict["nside-8"]["data"]
            ],
        },
        "nside-64": {
            "columns": COLUMNS_V0,
            "metadata": {"nside": 64},
            "data": [
                [
                    r[0],
                    r[1] * (1 + rtol_per_field["llh"]),
                    r[2] * (1 + rtol_per_field["E_in"]),
                    r[3] * (1 + rtol_per_field["E_tot"]),
                ]
                for r in alpha_pydict["nside-64"]["data"]
            ],
        },
    }
    beta = SkyScanResult.deserialize(beta_pydict)

    assert alpha.is_close(
        beta,
        equal_nan=True,
        dump_json_diff=json_diff,
        rtol_per_field=rtol_per_field,
    )
    assert beta.is_close(
        alpha,
        equal_nan=True,
        dump_json_diff=json_diff,
        rtol_per_field=rtol_per_field,
    )


def test_100(json_diff: Path) -> None:
    """Compare two simple instances with nans."""
    rtol_per_field = dict(llh=0.5, E_in=0.5, E_tot=0.5)

    alpha_pydict: PyDictResult = {
        "nside-8": {
            "columns": COLUMNS_V0,
            "metadata": {"nside": 8},
            "data": [
                [0, np.nan, 4643.5, 4736.5],
            ],
        },
    }
    alpha = SkyScanResult.deserialize(alpha_pydict)

    beta_pydict: PyDictResult = {
        "nside-8": {
            "columns": COLUMNS_V0,
            "metadata": {"nside": 8},
            "data": [
                [
                    r[0],
                    np.nan,
                    r[2] * (1 + rtol_per_field["E_in"]),
                    r[3] * (1 + rtol_per_field["E_tot"]),
                ]
                for r in alpha_pydict["nside-8"]["data"]
            ],
        },
    }
    beta = SkyScanResult.deserialize(beta_pydict)

    assert alpha.is_close(
        beta,
        equal_nan=True,
        dump_json_diff=json_diff,
        rtol_per_field=rtol_per_field,
    )
    assert beta.is_close(
        alpha,
        equal_nan=True,
        dump_json_diff=json_diff,
        rtol_per_field=rtol_per_field,
    )
    assert not alpha.is_close(
        beta,
        equal_nan=False,
        dump_json_diff=json_diff,
        rtol_per_field=rtol_per_field,
    )
    assert not beta.is_close(
        alpha,
        equal_nan=False,
        dump_json_diff=json_diff,
        rtol_per_field=rtol_per_field,
    )
