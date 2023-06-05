"""Test `is_close()`."""


from pathlib import Path
from typing import Any

import pytest
from skyreader import SkyScanResult
from skyreader.result import PyDictResult


def test_000(request: Any) -> None:
    """Compare same instances."""
    # rtol_per_field = dict(llh=0.5, E_in=0.5, E_tot=0.5)

    alpha_pydict: PyDictResult = {
        "nside-8": {
            "columns": ["index", "llh", "E_in", "E_tot"],
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
        dump_json_diff=Path(request.node.name + ".json"),
        do_disqualify_zero_energy_pixels=False,
    )


def test_010(request: Any) -> None:
    """Compare two simple instances."""
    rtol_per_field = dict(llh=0.5, E_in=0.5, E_tot=0.5)

    alpha_pydict: PyDictResult = {
        "nside-8": {
            "columns": ["index", "llh", "E_in", "E_tot"],
            "metadata": {"nside": 8},
            "data": [
                [0, 496.5, 4643.5, 4736.5],
            ],
        },
    }
    alpha = SkyScanResult.deserialize(alpha_pydict)

    beta_pydict: PyDictResult = {
        "nside-8": {
            "columns": ["index", "llh", "E_in", "E_tot"],
            "metadata": {"nside": 8},
            "data": [
                [
                    i[0],
                    i[1] * (1 + rtol_per_field["llh"]),
                    i[2] * (1 + rtol_per_field["E_in"]),
                    i[3] * (1 + rtol_per_field["E_tot"]),
                ]
                for i in alpha_pydict["nside-8"]["data"]
            ],
        },
    }
    beta = SkyScanResult.deserialize(beta_pydict)

    assert alpha.is_close(
        beta,
        equal_nan=True,
        dump_json_diff=Path(request.node.name + ".json"),
        do_disqualify_zero_energy_pixels=False,
        rtol_per_field=rtol_per_field,
    )
    assert beta.is_close(
        alpha,
        equal_nan=True,
        dump_json_diff=Path(request.node.name + ".json"),
        do_disqualify_zero_energy_pixels=False,
        rtol_per_field=rtol_per_field,
    )


@pytest.mark.parametrize("fail_field", ["llh", "E_in", "E_tot"])
def test_011__error(fail_field: str, request: Any) -> None:
    """Compare two simple instances."""
    rtol_per_field = dict(llh=0.5, E_in=0.5, E_tot=0.5)

    alpha_pydict: PyDictResult = {
        "nside-8": {
            "columns": ["index", "llh", "E_in", "E_tot"],
            "metadata": {"nside": 8},
            "data": [
                [0, 496.5, 4643.5, 4736.5],
            ],
        },
    }
    alpha = SkyScanResult.deserialize(alpha_pydict)

    # figure how to scale values to fail for alpha vs. BIGGER (not symmetrical)

    scale = dict(llh=1.0, E_in=1.0, E_tot=1.0)  # > 1/rtol should fail
    scale[fail_field] = (1 / rtol_per_field[fail_field]) * 1.1
    bigger_pydict: PyDictResult = {
        "nside-8": {
            "columns": ["index", "llh", "E_in", "E_tot"],
            "metadata": {"nside": 8},
            "data": [
                [
                    i[0],
                    i[1] * (1 + scale["llh"] * rtol_per_field["llh"]),
                    i[2] * (1 + scale["E_in"] * rtol_per_field["E_in"]),
                    i[3] * (1 + scale["E_tot"] * rtol_per_field["E_tot"]),
                ]
                for i in alpha_pydict["nside-8"]["data"]
            ],
        },
    }
    bigger = SkyScanResult.deserialize(bigger_pydict)
    assert not alpha.is_close(
        bigger,
        equal_nan=True,
        dump_json_diff=Path(request.node.name + ".json"),
        do_disqualify_zero_energy_pixels=False,
        rtol_per_field=rtol_per_field,
    )

    # figure how to scale values to fail for BIGGER vs. alpha (not symmetrical)

    scale = dict(llh=1.0, E_in=1.0, E_tot=1.0)  # >1 should fail
    scale[fail_field] = 2.0
    bigger_pydict = {
        "nside-8": {
            "columns": ["index", "llh", "E_in", "E_tot"],
            "metadata": {"nside": 8},
            "data": [
                [
                    i[0],
                    i[1] * (1 + scale["llh"] * rtol_per_field["llh"]),
                    i[2] * (1 + scale["E_in"] * rtol_per_field["E_in"]),
                    i[3] * (1 + scale["E_tot"] * rtol_per_field["E_tot"]),
                ]
                for i in alpha_pydict["nside-8"]["data"]
            ],
        },
    }
    bigger = SkyScanResult.deserialize(bigger_pydict)
    assert not bigger.is_close(
        alpha,
        equal_nan=True,
        dump_json_diff=Path(request.node.name + ".json"),
        do_disqualify_zero_energy_pixels=False,
        rtol_per_field=rtol_per_field,
    )


def test_020(request: Any) -> None:
    """Compare two multi-nside instances."""
    rtol_per_field = dict(llh=0.5, E_in=0.5, E_tot=0.5)

    alpha_pydict: PyDictResult = {
        "nside-8": {
            "columns": ["index", "llh", "E_in", "E_tot"],
            "metadata": {"nside": 8},
            "data": [
                [0, 496.5, 4643.5, 4736.5],
                [1, 586.5, 6845.5, 7546.5],
            ],
        },
        "nside-64": {
            "columns": ["index", "llh", "E_in", "E_tot"],
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
            "columns": ["index", "llh", "E_in", "E_tot"],
            "metadata": {"nside": 8},
            "data": [
                [
                    i[0],
                    i[1] * (1 + rtol_per_field["llh"]),
                    i[2] * (1 + rtol_per_field["E_in"]),
                    i[3] * (1 + rtol_per_field["E_tot"]),
                ]
                for i in alpha_pydict["nside-8"]["data"]
            ],
        },
        "nside-64": {
            "columns": ["index", "llh", "E_in", "E_tot"],
            "metadata": {"nside": 64},
            "data": [
                [
                    i[0],
                    i[1] * (1 + rtol_per_field["llh"]),
                    i[2] * (1 + rtol_per_field["E_in"]),
                    i[3] * (1 + rtol_per_field["E_tot"]),
                ]
                for i in alpha_pydict["nside-64"]["data"]
            ],
        },
    }
    beta = SkyScanResult.deserialize(beta_pydict)

    assert alpha.is_close(
        beta,
        equal_nan=True,
        dump_json_diff=Path(request.node.name + ".json"),
        do_disqualify_zero_energy_pixels=False,
        rtol_per_field=rtol_per_field,
    )
    assert beta.is_close(
        alpha,
        equal_nan=True,
        dump_json_diff=Path(request.node.name + ".json"),
        do_disqualify_zero_energy_pixels=False,
        rtol_per_field=rtol_per_field,
    )
