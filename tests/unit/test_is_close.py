"""Test `is_close()`."""


from skyreader import SkyScanResult
from skyreader.result import PyDictResult


def test_000() -> None:
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
        dump_json_diff=None,
        do_disqualify_zero_energy_pixels=False,
    )


def test_001() -> None:
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
                    0,
                    alpha_pydict["nside-8"]["data"][0][1] * (1 + rtol_per_field["llh"]),
                    alpha_pydict["nside-8"]["data"][0][2]
                    * (1 + rtol_per_field["E_in"]),
                    alpha_pydict["nside-8"]["data"][0][3]
                    * (1 + rtol_per_field["E_tot"]),
                ],
            ],
        },
    }
    beta = SkyScanResult.deserialize(beta_pydict)

    assert alpha.is_close(
        beta,
        equal_nan=True,
        dump_json_diff=None,
        do_disqualify_zero_energy_pixels=False,
        rtol_per_field=rtol_per_field,
    )
    assert beta.is_close(
        alpha,
        equal_nan=True,
        dump_json_diff=None,
        do_disqualify_zero_energy_pixels=False,
        rtol_per_field=rtol_per_field,
    )
