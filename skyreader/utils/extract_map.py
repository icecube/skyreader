from typing import Union
import logging

import healpy
import numpy as np

from ..result import SkyScanResult

LOGGER = logging.getLogger("skyreader.extract_map")


def extract_map(
        result: SkyScanResult,
        llh_map: bool = True,
        angular_error_floor: Union[None, float] = None,
):
    """
    Extract from the output of skymap_scanner the healpy map
    args:
        - result: SkyScanResult. The output of the Skymap Scanner
        - llh_map: bool = True. If True the likelihood will be plotted,
            otherwise the probability
        - angular_error_floor: Union[None, float] = None. if not None,
            sigma of the gaussian to convolute the map with in deg.

    returns:
        - grid_value: value-per-scanned-pixel (pixels with
            different nsides)
        - grid_ra: right ascension for each pixel in grid_value
        - grid_dec: declination for each pixel in grid_value
        - equatorial_map: healpix map with maximum nside (all pixels
            with same nside)
    """

    grid_map = dict()

    nsides = result.nsides
    max_nside = max(nsides)
    equatorial_map = np.full(healpy.nside2npix(max_nside), np.nan)

    for nside in nsides:
        LOGGER.info(f"constructing map for nside {nside}...")
        npix = healpy.nside2npix(nside)
        map_data = result.result[f'nside-{nside}']
        pixels = map_data['index']
        values = map_data['llh']
        this_map = np.full(npix, np.nan)
        this_map[pixels] = values
        if nside < max_nside:
            this_map = healpy.ud_grade(this_map, max_nside)
        mask = np.logical_and(~np.isnan(this_map), np.isfinite(this_map))
        equatorial_map[mask] = this_map[mask]

        for pixel_data in result.result[f"nside-{nside}"]:
            pixel = pixel_data['index']
            value = pixel_data['llh']
            if np.isfinite(value) and not np.isnan(value):
                tmp_theta, tmp_phi = healpy.pix2ang(nside, pixel)
                tmp_dec = np.pi/2 - tmp_theta
                tmp_ra = tmp_phi
                grid_map[(tmp_dec, tmp_ra)] = value
                # nested_pixel = healpy.ang2pix(
                #     nside, tmp_theta, tmp_phi, nest=True
                # )
                # uniq = 4*nside*nside + nested_pixel
                # uniq_list.append(uniq)
        LOGGER.info(f"done with map for nside {nside}...")

    grid_dec_list, grid_ra_list, grid_value_list = [], [], []

    for (dec, ra), value in grid_map.items():
        grid_dec_list.append(dec)
        grid_ra_list.append(ra)
        grid_value_list.append(value)
    grid_dec: np.ndarray = np.asarray(grid_dec_list)
    grid_ra: np.ndarray = np.asarray(grid_ra_list)
    grid_value: np.ndarray = np.asarray(grid_value_list)
    # uniq_array: np.ndarray = np.asarray(uniq_list)

    sorting_indices = np.argsort(grid_value)
    grid_value = grid_value[sorting_indices]
    grid_dec = grid_dec[sorting_indices]
    grid_ra = grid_ra[sorting_indices]
    # uniq_array = uniq_array[sorting_indices]

    min_value = grid_value[0]

    # renormalize
    grid_value = grid_value - min_value
    min_value = 0.

    # renormalize
    equatorial_map[np.isinf(equatorial_map)] = np.nan
    equatorial_map -= np.nanmin(equatorial_map)

    if llh_map:
        # show 2 * delta_LLH
        grid_value = grid_value * 2.
        equatorial_map *= 2.
    else:
        # Convert to probability
        equatorial_map = np.exp(-1. * equatorial_map)
        equatorial_map = equatorial_map / np.nansum(equatorial_map)

        # nan values are a problem for the convolution and the contours
        min_map = np.nanmin(equatorial_map)
        equatorial_map[np.isnan(equatorial_map)] = min_map

        if angular_error_floor is not None:
            # convolute with a gaussian. angular_error_floor is the
            # sigma in deg.
            equatorial_map = healpy.smoothing(
                equatorial_map,
                sigma=np.deg2rad(angular_error_floor),
            )

            # normalize map
            min_map = np.nanmin(equatorial_map[equatorial_map >= 0.0])
            equatorial_map[np.isnan(equatorial_map)] = min_map
            equatorial_map = equatorial_map.clip(min_map, None)
            normalization = np.nansum(equatorial_map)
            equatorial_map = equatorial_map / normalization

        # obtain values for grid map
        grid_value = healpy.get_interp_val(
            equatorial_map, np.pi/2 - grid_dec, grid_ra
        )
        grid_value[np.isnan(grid_value)] = min_map
        grid_value = grid_value.clip(min_map, None)

    return grid_value, grid_ra, grid_dec, equatorial_map
