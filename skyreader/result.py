"""For encapsulating the results of an event scan in a single instance."""

# fmt: off
# pylint: skip-file
# flake8: noqa

import itertools as it
import json
import logging
import pickle
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple, TypedDict, Union

import healpy  # type: ignore[import]
import matplotlib  # type: ignore[import]
import meander  # type: ignore[import]
import numpy as np
import pandas as pd  # type: ignore[import]
from astropy.io import ascii  # type: ignore[import]
from matplotlib import patheffects
from matplotlib import pyplot as plt
from matplotlib import text

from .event_metadata import EventMetadata

###############################################################################
# CONSTANTS


# bookkeeping for comparing values
DEFAULT_RTOL_PER_FIELD = {  # w/ rtol values
    # any field not here is assumed to require '==' for comparison
    "llh": 1e-4,
    "E_in": 1e-2,
    "E_tot": 1e-2,
    "X": 1e-3,
    "Y": 1e-3,
    "Z": 1e-3,
    "T": 1e-3,
}
ZERO_MAKES_FIELD_ALWAYS_ISCLOSE = [
    # if a pixel field's val is 0, then that datapoint is "isclose" to any value
    "E_in",
    "E_tot",
]


###############################################################################
# DATA TYPES


class PyDictNSidePixels(TypedDict):
    columns: List[str]
    metadata: Dict[str, Any]
    data: List[List[Union[int, float]]]


PyDictResult = Dict[str, PyDictNSidePixels]


###############################################################################
# MAIN CLASS

class SkyScanResult:
    """This class parses a scan result and stores the relevant numeric results
    of the scan. Ideally it should serve as the basic data structure for
    plotting / processing / transmission of the scan result.

    `result` is a dictionary keyed by 'nside: str' values for which a scan
    result is available (e.g. 8, 64, 512).

    The scan result is a dictionary:
    - i (pixel index, integer) ->
        'frame', 'llh', 'recoLossesInside', 'recoLossesTotal'

    The numeric values of interest are 'llh', 'recoLossesInside',
    'recoLossesTotal'. The pixel indices in the input dictionary are in
    general unsorted (python dict are unsorted by design) and are
    incomplete (since fine-grained scans only cover a portion of the
    HEALPIX area). The class stores the each result in a np
    structured array sorted by the pixel index, which is stored in a
    dedicated field.
    """

    # versioned dtypes
    PIXEL_TYPES = {0: np.dtype([("index", int), ("llh", float), ("E_in", float), ("E_tot", float),]),
                   1: np.dtype([("index", int), ("llh", float), ("E_in", float), ("E_tot", float),
                                ("X", float), ("Y", float), ("Z", float), ("T", float),])}
    ATOL = 1.0e-8  # 1.0e-8 is the default used by np.isclose()

    MINIMAL_METADATA_FIELDS: Final[List[str]] = "run_id event_id mjd event_type nside".split()

    def __init__(self, result: Dict[str, np.ndarray]):
        self.logger = logging.getLogger(__name__)
        self.result = result
        self.nsides = sorted([self.parse_nside(key) for key in self.result])
        self.pixel_type = self.PIXEL_TYPES[self.get_event_metadata().version]
        self.pixel_fields: Tuple[str, ...] = self.pixel_type.names if self.pixel_type.names is not None else tuple()

        # validate result data
        if not isinstance(result, dict):
            raise ValueError("'result' must be an instance of Dict[str, np.ndarray]")
        for nside in result:
            try:
                self.parse_nside(nside)
            except (KeyError, ValueError) as e:
                raise ValueError(f"'result' has invalid nside key: {nside}") from e
            if not isinstance(result[nside], np.ndarray):
                raise ValueError("'result' must be an instance of Dict[str, np.ndarray]")
            if result[nside].dtype != self.pixel_type:
                raise ValueError(
                    f"'result' has invalid dtype {result[nside].dtype} "
                    f"should be {self.pixel_type} "
                )

        self.logger.debug(f"Metadata for this result: {[self.result[_].dtype.metadata for _ in self.result]}")



    """
    Comparison operators and methods
    """

    def __eq__(self, other: object) -> bool:
        """Are the two instance's result lists strictly equal?"""
        if not isinstance(other, SkyScanResult):
            return False
        if self.result.keys() != other.result.keys():
            return False
        # NOTE: will return false if NaN are present
        # np.array_equal() supports `equal_nan` option only from version 1.19
        return all(
            np.array_equal(self.result[nside], other.result[nside])
            for nside in self.result
        )

    def isclose_datapoint(
        self,
        s_val: float,
        o_val: float,
        field: str,
        equal_nan: bool,
        rtol_per_field: Dict[str, float],
    ) -> Tuple[float, bool]:
        """Get the diff float-value and test truth-value for the 2 pixel
        datapoints."""
        if field not in rtol_per_field:
            raise ValueError(
                f"Datapoint field ({field}) cannot be compared by "
                f"'is_close_datapoint()', must use '=='"
            )
        if field in ZERO_MAKES_FIELD_ALWAYS_ISCLOSE and (s_val == 0.0 or o_val == 0.0):
            return float("nan"), True
        try:
            rdiff = (abs(s_val - o_val) - self.ATOL) / abs(o_val)  # used by np.isclose
        except ZeroDivisionError:
            rdiff = float("inf")

        return (
            rdiff,
            bool(
                np.isclose(
                    s_val,
                    o_val,
                    equal_nan=equal_nan,
                    rtol=rtol_per_field[field],
                    atol=self.ATOL,
                )
            ),
        )

    def isclose_pixel(
        self,
        sre_pix: np.ndarray,
        ore_pix: np.ndarray,
        equal_nan: bool,
        rtol_per_field: Dict[str, float],
        fields_to_compare: Tuple[str, ...],
    ) -> Tuple[List[float], List[bool]]:
        """Get the diff float-values and test truth-values for the 2 pixel-
        data.

        The datapoints are compared face-to-face (zipped).
        """
        diff_vals = []
        test_vals = []

        for s_val, o_val, field in zip(sre_pix, ore_pix, fields_to_compare):
            s_val, o_val = float(s_val), float(o_val)

            # CASE: a "require close" datapoint
            if field in rtol_per_field:
                diff, test = self.isclose_datapoint(s_val, o_val, field, equal_nan, rtol_per_field)
            # CASE: a "require equal" datapoint
            else:
                diff, test = s_val - o_val, s_val == o_val

            diff_vals.append(diff)
            test_vals.append(test)

        return diff_vals, test_vals

    def has_minimal_metadata(self) -> bool:
        """Check that the minimum metadata is set."""
        if len(self.result) == 0:
            return False

        for mk in self.MINIMAL_METADATA_FIELDS:
            for k in self.result:
                if self.result[k].dtype.metadata is None:
                    return False
                if mk not in self.result[k].dtype.metadata:
                    return False
        return True

    def get_event_metadata(self) -> EventMetadata:
        """Get the EventMetadata portion of the result's metadata."""
        if self.has_minimal_metadata():
            first_metadata = self.result[list(self.result.keys())[0]].dtype.metadata
            return EventMetadata(
                first_metadata['run_id'],
                first_metadata['event_id'],
                first_metadata['event_type'],
                first_metadata['mjd'],
                first_metadata.get('is_real_event', False),  # assume simulated event
                first_metadata.get('version', 0),  # fallback to version 0 if not set
            )
        else:
            self.logger.warning("Metadata doesn't seem to exist and will not be used for plotting.")
            return EventMetadata(0, 0, '', 0, False, 0)
        
    def get_results_per_nside(self, nside: int) -> np.ndarray:
        "get the results for the pixels at a given nside"
        return self.result[f"nside-{nside}"]

    def isclose_nside(self,
        other: "SkyScanResult",
        equal_nan: bool,
        rtol_per_field: Dict[str, float],
        nside: str,
    ) -> Tuple[bool, List[Tuple[Tuple[Any, ...], Tuple[Any, ...], Tuple[float, ...], Tuple[bool, ...]]]]:
        """Get whether the two nside's pixels are all "close"."""
        # zip-iterate each pixel-data
        nside_diffs = []
        fields_to_compare = tuple(set(self.pixel_fields).intersection(other.pixel_fields))

        fillarr = np.full((len(fields_to_compare),), np.nan,
                          dtype=[(_, float) for _ in fields_to_compare])
        sre = self.result.get(nside, fillarr)
        ore = other.result.get(nside, fillarr)
        for sre_pix, ore_pix in it.zip_longest(sre[list(fields_to_compare)],
                                               ore[list(fields_to_compare)],
                                               fillvalue=fillarr):
            diff_vals, test_vals = self.isclose_pixel(
                sre_pix, ore_pix, equal_nan, rtol_per_field, fields_to_compare # type: ignore[arg-type]
            )
            pix_diff = (
                tuple(sre_pix.tolist()),
                tuple(ore_pix.tolist()),
                tuple(diff_vals),  # diff float-value
                tuple(test_vals),  # test truth-value
            )
            for vals in pix_diff:
                self.logger.debug(f"{nside}: {vals}")
            nside_diffs.append(pix_diff)

        # aggregate test-truth values
        nside_equal = {
            field: all(d[3][fields_to_compare.index(field)] for d in nside_diffs)
            for field in set(fields_to_compare) - set(rtol_per_field)
        }

        nside_close = {
            field: all(d[3][fields_to_compare.index(field)] for d in nside_diffs)
            for field in fields_to_compare
        }

        # log results (test-truth values)
        if not all(nside_equal.values()):
            self.logger.info(f"Mismatched pixel indices for nside={nside}")
        if not all(nside_close.values()):
            self.logger.info(f"Mismatched numerical results for nside={nside}")
            self.logger.debug(f"{nside_close}")

        return all(nside_equal.values()) and all(nside_close.values()), nside_diffs

    def is_close(
        self,
        other: "SkyScanResult",
        equal_nan: bool = True,
        dump_json_diff: Optional[Path] = None,
        rtol_per_field: Optional[Dict[str, float]] = None,
    ) -> bool:
        """Checks if two results are close by requiring strict equality on
        pixel indices and close condition on numeric results.

        Args:
            `other`
                the instance to compare
            `equal_nan`
                whether to let `nan == nan` be True
                (default: `True`)
            `dump_json_diff`
                get a json file containing every comparison at the pixel-data level
                (default: `None`)
            `rtol_per_field`
                a mapping of each field to a rtol value
                (default: `DEFAULT_RTOL_PER_FIELD`)

        Returns:
            bool: True if `other` and `self` are close
        """
        if rtol_per_field is None:
            rtol_per_field = DEFAULT_RTOL_PER_FIELD

        close: Dict[str, bool] = {}  # one bool for each nside value
        diffs: Dict[str, list] = {}  # (~4x size of self.results) w/ per-pixel info

        # now check individual nside-iterations
        for nside in sorted(self.result.keys() & other.result.keys(), reverse=True):
            self.logger.info(f"Comparing for nside={nside}")
            # Q: why aren't we using np.array_equal and np.allclose?
            # A: we want detailed pixel-level diffs w/out repeating detailed code
            close[nside], diffs[nside] = self.isclose_nside(
                other, equal_nan, rtol_per_field, nside
            )

        # finish up
        self.logger.info(f"Comparison result: {close}")

        if dump_json_diff:
            with open(dump_json_diff, "w") as f:
                self.logger.info(f"Writing diff to {dump_json_diff}...")
                json.dump(diffs, f, indent=3)

        return all(close.values())

    """
    Auxiliary methods
    """

    @staticmethod
    def format_nside(nside) -> str:
        return f"nside-{nside}"

    @staticmethod
    def parse_nside(key) -> int:
        return int(key.split("nside-")[1])

    def get_nside_string(self) -> str:
        """Returns a string string listing the nside values to be included in
        the output filename."""
        # keys have a 'nside-NNN' format but we just want to extract the nside values to build the string
        # parsing back and forth numbers to strings is not the most elegant choice but works for now
        # TODO: possibly better to use integer values as keys in self.result
        return "_".join([str(nside) for nside in self.nsides])

    def get_filename(
        self,
        event_metadata: EventMetadata,
        extension: str,
        output_dir: Union[str, Path, None] = None
    ) -> Path:
        """Make a filepath for writing representations of `self` to disk."""
        if not extension.startswith('.'):
            extension = '.' + extension

        if nside_string := self.get_nside_string():
            filename = Path(f"{str(event_metadata)}_{nside_string}{extension}")
        else:
            raise ValueError("cannot create filename for an empty result")

        if output_dir is not None:
            filename = output_dir / Path(filename)
        return filename

    """
    NPZ input / output
    """

    @classmethod
    def read_npz(cls, filename: Union[str, Path]) -> "SkyScanResult":
        """Load from .npz file."""
        npz = np.load(filename)
        result = dict()
        if "header" not in npz:
            for key in npz.keys():
                result[key] = npz[key]
        else:
            h = npz["header"]
            for v in h:
                key = cls.format_nside(v['nside'])
                _dtype = np.dtype(
                    npz[key].dtype,
                    metadata={k:value for k, value in zip(h.dtype.fields.keys(), v)},  # type: ignore[call-overload]
                )
                result[key] = np.array(list(npz[key]), dtype=_dtype)
        return cls(result=result)

    def to_npz(
        self,
        event_metadata: EventMetadata,
        output_dir: Union[str, Path, None] = None,
    ) -> Path:
        """Save to .npz file."""
        filename = self.get_filename(event_metadata, '.npz', output_dir)

        try:
            first = next(iter(self.result.values()))
        except StopIteration: # no results yet
            np.savez(filename, **self.result)  # type: ignore
            return Path(filename)

        try:
            metadata_dtype = np.dtype(
                [
                    (k, type(v)) if not isinstance(v, str) else (k, f"U{len(v)}")
                    for k, v in first.dtype.metadata.items()
                ],
            )
            header = np.array(
                [
                    tuple(self.result[k].dtype.metadata[mk] for mk in metadata_dtype.fields)  # type: ignore[union-attr]
                    for k in self.result
                ],
                dtype=metadata_dtype,
            )
            np.savez(filename, header=header, **self.result)  # type: ignore
        except (TypeError, AttributeError):
            np.savez(filename, **self.result)  # type: ignore

        return Path(filename)

    """
    JSON input / output
    """

    @classmethod
    def read_json(cls, filename: Union[str, Path]) -> "SkyScanResult":
        """Load from .json file."""
        with open(filename) as f:
            pydict = json.load(f)
        return cls.deserialize(pydict)

    def to_json(
        self,
        event_metadata: EventMetadata,
        output_dir: Union[str, Path, None] = None
    ) -> Path:
        """Save to .json file."""
        filename = self.get_filename(event_metadata, '.json', output_dir)
        pydict = self.serialize()
        with open(filename, 'w') as f:
            json.dump(pydict, f, indent=4)
        return filename

    """
    Serialize/deserialize (input / output)
    """

    @classmethod
    def deserialize(cls, pydict: PyDictResult) -> "SkyScanResult":
        """Deserialize from a python-native dict."""
        result = dict()

        for nside, pydict_nside_pixels in pydict.items():
            metadata=pydict_nside_pixels['metadata']  # type: ignore[call-overload]
            data_version = metadata.get('version', 0)
            pixel_type = cls.PIXEL_TYPES[data_version]
            pixel_fields: Tuple[str, ...] = pixel_type.names # type: ignore[assignment]
            # validate keys
            if set(pydict_nside_pixels.keys()) != {'columns', 'metadata', 'data'}:
                raise ValueError(f"PyDictResult entry has extra/missing keys: {pydict_nside_pixels.keys()}")

            # check 'columns'
            if pydict_nside_pixels['columns'] != list(pixel_fields):
                raise ValueError(
                    f"PyDictResult entry has invalid 'columns' entry "
                    f"({pydict_nside_pixels['columns']}) should be {list(pixel_fields)}"
                )

            # check 'metadata'
            try:
                if metadata['nside'] != cls.parse_nside(nside):
                    raise ValueError(
                        f"PyDictResult entry has incorrect 'metadata'.'nside' value: "
                        f"{pydict_nside_pixels['metadata']['nside']} should be {cls.parse_nside(nside)}"
                    )
            except (KeyError, TypeError) as e:
                raise ValueError("PyDictResult entry has missing key 'nside'") from e

            # read/convert
            _dtype = np.dtype(
                pixel_type, metadata=metadata
            )
            result_nside_pixels = np.zeros(len(pydict_nside_pixels['data']), dtype=_dtype)
            for i, pix_4list in enumerate(sorted(pydict_nside_pixels['data'], key=lambda x: x[0])):
                result_nside_pixels[i] = tuple(pix_4list)

            result[nside] = result_nside_pixels

        return cls(result)

    def serialize(self) -> PyDictResult:
        """Serialize as a python-native dict.

        Example:
        {
            'nside-8': {
                "columns": [
                    "index",
                    "llh",
                    "E_in",
                    "E_tot"
                ],
                "metadata": {
                    "nside": 8,
                    ...
                }
                "data": [
                    [
                        0,
                        496.81227052,
                        4643.8910975498,
                        4736.3116335241
                    ],
                    [
                        1,
                        503.6851841852,
                        5058.9879730721,
                        585792.3192455448
                    ],
                    ...
                ]
            },
            ...
        }
        """
        pydict: PyDictResult = {}
        for nside in self.result:
            nside_data: np.ndarray = self.result[nside]
            df = pd.DataFrame(
                nside_data,
                columns=list(nside_data.dtype.names),
            )
            pydict[nside] = {k:v for k,v in df.to_dict(orient='split').items() if k != 'index'}  # type: ignore[assignment]
            pydict[nside]['metadata'] = dict()

            for key in nside_data.dtype.metadata:
                # dtype.metadata is a mappingproxy (dict-like) containing numpy-typed values
                # convert numpy types to python bultins to be JSON-friendly
                val = nside_data.dtype.metadata[key]
                if isinstance(val, np.generic):
                    # numpy type, non serializable
                    # convert to python built-in by calling item()
                    pydict[nside]['metadata'][key] = nside_data.dtype.metadata[key].item()
                else:
                    # likely a natively serializable python built-in 
                    pydict[nside]['metadata'][key] = val
        return pydict

    """
    Querying
    """

    def llh(self, ra, dec):
        for nside in self.nsides[::-1]:
            grid_pix = healpy.ang2pix(nside, np.pi/2 - dec, ra)
            _res = self.result[self.format_nside(nside)]
            llh = _res[_res['index']==grid_pix]['llh']
            if llh.size > 0:
                return llh

    @property
    def min_llh(self):
        return self.best_fit['llh']

    @cached_property
    def best_fit(self):
        _minllh = np.inf
        for k in self.result:
            _res = self.result[k]
            _min = _res['llh'].min()
            if _min < _minllh:
                _minllh = _min
                _bestfit = _res[_res['llh'].argmin()]
        return _bestfit

    @property
    def best_dir(self):
        minCoDec, minRA = healpy.pix2ang(self.best_fit.dtype.metadata['nside'], self.best_fit['index'])
        minDec = np.pi/2 - minCoDec
        return minRA, minDec
