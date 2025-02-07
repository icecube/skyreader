from typing import Union, Tuple, List
import logging

import healpy  # type: ignore[import]
import mhealpy  # type: ignore[import]
import numpy as np

from ..result import SkyScanResult

LOGGER = logging.getLogger("skyreader.extract_map")


def extract_map(
        result: SkyScanResult,
        llh_map: bool = True,
        angular_error_floor: Union[None, float] = None,
        remove_min_val: bool = True,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Extract from the output of skymap_scanner the healpy map
    args:
        - result: SkyScanResult. The output of the Skymap Scanner
        - llh_map: bool = True. If True the likelihood will be plotted,
            otherwise the probability
        - angular_error_floor: Union[None, float] = None. if not None,
            sigma of the gaussian to convolute the map with in deg.
        - remove_min_val: bool = True. Remove minimum value from -llh
          no effect if probability map.

    returns:
        - grid_value: value-per-scanned-pixel (pixels with
            different nsides)
        - grid_ra: right ascension for each pixel in grid_value
        - grid_dec: declination for each pixel in grid_value
        - equatorial_map: healpix map with maximum nside (all pixels
            with same nside)
        - uniq_array: uniqs for all the pixels of the map
    """

    grid_map = dict()

    nsides = result.nsides
    max_nside = max(nsides)
    equatorial_map = np.full(healpy.nside2npix(max_nside), np.nan)
    uniq_list = []

    for nside in nsides:
        LOGGER.info(f"constructing map for nside {nside}...")
        npix = healpy.nside2npix(nside)
        map_data = result.get_results_per_nside(nside)
        pixels = map_data['index']
        values = map_data['llh']
        this_map = np.full(npix, np.nan)
        this_map[pixels] = values
        if nside < max_nside:
            this_map = healpy.ud_grade(this_map, max_nside)
        mask = np.logical_and(~np.isnan(this_map), np.isfinite(this_map))
        equatorial_map[mask] = this_map[mask]

        for pixel_data in result.get_results_per_nside(nside):
            pixel = pixel_data['index']
            value = pixel_data['llh']
            nested_pixel = healpy.ring2nest(nside, pixel)
            uniq = 4*nside*nside + nested_pixel
            uniq_list.append(uniq)
            tmp_theta, tmp_phi = healpy.pix2ang(nside, pixel)
            tmp_dec = np.pi/2 - tmp_theta
            tmp_ra = tmp_phi
            grid_map[(tmp_dec, tmp_ra)] = value

        # In case of pointed scans, it helps filling the first nside
        # with empty pixels (especially for saving the multiorder map)
        if nside == nsides[0]:
            grid_map, uniq_list = _fill_first_nside_empty(
                nside, result, grid_map, uniq_list
            )

        LOGGER.info(f"done with map for nside {nside}...")

    grid_dec_list, grid_ra_list, grid_value_list = [], [], []

    for (dec, ra), value in grid_map.items():
        grid_dec_list.append(dec)
        grid_ra_list.append(ra)
        grid_value_list.append(value)
    grid_dec: np.ndarray = np.asarray(grid_dec_list)
    grid_ra: np.ndarray = np.asarray(grid_ra_list)
    grid_value: np.ndarray = np.asarray(grid_value_list)
    uniq_array: np.ndarray = np.asarray(uniq_list)

    sorting_indices = np.argsort(grid_value)
    grid_value = grid_value[sorting_indices]
    grid_dec = grid_dec[sorting_indices]
    grid_ra = grid_ra[sorting_indices]
    uniq_array = uniq_array[sorting_indices]

    min_value = grid_value[0]

    if remove_min_val or (not llh_map):
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
        min_map = np.nanmin(equatorial_map)

        if angular_error_floor is not None:
            # convolute with a gaussian. angular_error_floor is the
            # sigma in deg.
            # nan values are a problem for the convolution and the contours
            equatorial_map = healpy.smoothing(
                np.nan_to_num(equatorial_map),
                sigma=np.deg2rad(angular_error_floor),
            )

            # normalize map
            min_map = np.nanmin(equatorial_map[equatorial_map >= 0.])
            equatorial_map = equatorial_map.clip(min_map, None)
            normalization = np.nansum(equatorial_map)
            equatorial_map = equatorial_map / normalization

        # obtain values for grid map
        grid_value = healpy.get_interp_val(
            equatorial_map, np.pi/2 - grid_dec, grid_ra
        )
        grid_value = grid_value.clip(min_map, None)

    return grid_value, grid_ra, grid_dec, equatorial_map, uniq_array


def _fill_first_nside_empty(
    nside: int,
    result: SkyScanResult,
    grid_map: dict,
    uniq_list: list,
):
    """
    Fill the grid_map at the first nside with empty pixels
    """
    tot_npix = healpy.nside2npix(nside)
    if tot_npix > len(result.get_results_per_nside(nside)):
        ring_pixels = np.arange(tot_npix)
        nest_pixels = healpy.ring2nest(nside, ring_pixels)
        uniq_pixels = mhealpy.nest2uniq(nside, nest_pixels)
        for uni, rin in zip(uniq_pixels, ring_pixels):
            if uni not in uniq_list:
                uniq_list.append(uni)
                tmp_theta, tmp_phi = healpy.pix2ang(nside, rin)
                tmp_dec = np.pi/2 - tmp_theta
                tmp_ra = tmp_phi
                grid_map[(tmp_dec, tmp_ra)] = np.nan
    return grid_map, uniq_list


def get_contour_levels(
    equatorial_map: np.ndarray,
    llh_map: bool = True,
    systematics: bool = False,
):
    """
    get contour levels for the desired map

    args:
        - equatorial_map: np.ndarray. Necessary in case of
            probability map.
        - llh_map: bool. If True llh levels, otherwise probability
            levels.
        - systematics: bool. Only for llh maps. If True include
            recalibrated llh values from Pan-Starrs event 127852
            (IC160427A syst.)

    returns:
        - contour_levels, levels of the contours
        - contour_labels, respective labels for the contours
        - contour_colors, respective colors for the contours
    """

    # Calculate the contour levels
    if llh_map:  # likelihood map
        min_value = np.nanmin(equatorial_map)
        if systematics:
            # from Pan-Starrs event 127852
            # these are values determined from MC by Will on the TS (2*LLH)
            # Not clear yet how to translate this for the probability map
            contour_levels = (np.array([22.2, 64.2])+min_value)
            contour_labels = [
                r'50% (IC160427A syst.)', r'90% (IC160427A syst.)'
            ]
            contour_colors = ['k', 'r']
        # Wilks
        else:
            contour_levels = (
                np.array([1.39, 4.61, 11.83, 28.74])+min_value
            )[:3]
            contour_labels = [
                r'50%', r'90%', r'3$\sigma$', r'5$\sigma$'
            ][:3]
            contour_colors = ['k', 'r', 'g', 'b'][:3]
    else:  # probability map
        if systematics:
            raise AssertionError(
                "No corrected values for contours in probability maps"
            )
        else:
            sorted_values = np.sort(equatorial_map)[::-1]
            probability_levels = (
                np.array([0.5, 0.9, 1-1.35e-3, 1-2.87e-7])
            )[:3]
            contour_levels = []
            for prob in probability_levels:
                level_index = (
                    np.nancumsum(sorted_values) >= prob
                ).tolist().index(True)
                level = sorted_values[level_index]
                contour_levels.append(level)
            contour_labels = [r'50%', r'90%', r'3$\sigma$', r'5$\sigma$'][:3]
            contour_colors = ['k', 'r', 'g', 'b'][:3]

    return contour_levels, contour_labels, contour_colors


def find_pixels_double_nside(
    nside: int,
    indexes: np.ndarray,
):
    """
    Given indexes of pixels at a given nside, find which are
    the pixels inside these pixels with a double nside

    args:
        - nside: int. nside of the pixels to investigate.
        - indexes: np.ndarray. indexes on ring ordering of the pixels
            to investigate.

    returns:
        - idxs_inside: np.ndarray, array containing an array for each
            initial pixel which tells which are the pixels with double
            nside inside that pixel. The pixels are in ring ordering.
    """

    vecs = healpy.boundaries(nside, indexes)
    transposed_vecs = np.transpose(vecs, axes=(0,2,1))
    idxs_inside = np.array(
        [healpy.query_polygon(nside*2, vs) for vs in transposed_vecs]
    )
    return idxs_inside


def already_filled_uniqs_for_nside(
    nside: int,
    next_nside: int,
    uniqs: np.ndarray
):
    """
    Check if among the input uniqs there are pixels at a finer nside
    which are inside other pixels with a bigger nside

    args:
        -nside: int. nside of the pixels to see if there are finer pixels
            inside.
        -next_nside: int. nside of the finer pixes which we want to see
            if they are inside the coarser pixels
        -uniqs: np.ndarray. Uniqs for all the pixels in the map

    returns:
        already_filled_uniqs: np.ndarray. Uniqs of the coarser pixels
            which are already filled.
    """
    nside_per_pixel = mhealpy.uniq2nside(uniqs)
    uniqs_nside = uniqs[nside_per_pixel == nside]
    uniqs_next_nside = uniqs[nside_per_pixel == next_nside]
    pixels_nside = healpy.nest2ring(
        nside, mhealpy.uniq2nest(uniqs_nside)[1]
    )
    idxs_double_nside = find_pixels_double_nside(nside, pixels_nside)
    already_filled_uniqs = []
    for uniq_original_nside, idxs_pixel in zip(
        uniqs_nside, idxs_double_nside
    ):
        progressive_nside = nside*2
        while progressive_nside != next_nside:
            idxs_pixel = np.concatenate(
                find_pixels_double_nside(progressive_nside, idxs_pixel)
            )
            progressive_nside *= 2
        uniqs_pixel = mhealpy.nest2uniq(
            next_nside, healpy.ring2nest(next_nside, idxs_pixel)
        )
        pixel_already_filled = False
        for uniq in uniqs_next_nside:
            if uniq in uniqs_pixel:
                pixel_already_filled = True
                continue
        if pixel_already_filled:
            already_filled_uniqs.append(uniq_original_nside)
    return np.array(already_filled_uniqs)


def find_filled_pixels(uniqs: np.ndarray):
    """
    given an array of uniqs, finds which pixels already have finer
    pixels inside and returns array with the indeces of these pixels

    args:
        - uniqs: np.ndarray. Array of uniqs per each pixel of the map

    returns:
        - already_filled_indeces: np.ndarray. Array with the indeces
            of the already_filled_uniqs in the input array
    """
    nside_per_pixel = mhealpy.uniq2nside(uniqs)
    nsides = np.unique(nside_per_pixel)
    already_filled_uniqs = []
    for nside_index, nside in enumerate(nsides[:-1]):
        next_nside = nsides[nside_index + 1]
        already_filled_uniqs_nside = already_filled_uniqs_for_nside(
            nside, next_nside, uniqs
        )
        already_filled_uniqs.append(already_filled_uniqs_nside)
    already_filled_uniqs = np.concatenate(already_filled_uniqs)
    already_filled_indeces = np.array(
        [np.where(uniqs == uni)[0][0] for uni in already_filled_uniqs]
    )
    return already_filled_indeces


def clean_data_multiorder_map(
    grid_value: np.ndarray, uniqs: np.ndarray
):
    """
    Clean a map from the pixels which have a finer scan inside

    args:
        - grid_value: np.ndarray. Value of probability (likelihood)
            per each scanned pixel.
        - uniqs: np.ndarray. Uniqs per each scanned pixel to univocally
            identify them

    returns:
        - grid_value: np.ndarray. Cleaned array
        - uniqs: np.ndarray. Cleaned array
    """
    filled_indeces = find_filled_pixels(uniqs)
    grid_value = np.delete(grid_value, filled_indeces)
    uniqs = np.delete(uniqs, filled_indeces)
    return grid_value, uniqs


def prepare_flattened_map(
    equatorial_map: np.ndarray,
    llh_map: bool,
) -> Tuple[np.ndarray, List[str], Union[List[str], None]]:
    """
    Create the healpix map that needs to be saved keeping
    into account if it is a probability or a llh map
    """
    if llh_map:
        column_names = ['2DLLH']
        column_units = None
    else:
        # avoid excessively heavy data format for the flattened map
        equatorial_map[equatorial_map < 1e-16] = np.nanmean(
            equatorial_map[equatorial_map < 1e-16]
        )
        column_names = ["PROB"]
        column_units = ["pix-1"]
    return equatorial_map, column_names, column_units


def prepare_multiorder_map(
    grid_value: np.ndarray,
    uniq_array: np.ndarray,
    llh_map: bool,
    column_names: List[str]
) -> Tuple[mhealpy.HealpixMap, List[str]]:
    """
    Create the mhealpix map that needs to be saved keeping
    into account if it is a probability or a llh map
    """
    # clean from redundant pixels
    grid_value, uniq_array = clean_data_multiorder_map(
        grid_value, uniq_array
    )
    # save multiorder version of the map
    if llh_map:
        multiorder_map = mhealpy.HealpixMap(grid_value, uniq_array)
    else:
        all_nsides = mhealpy.uniq2nside(uniq_array)
        max_nside = np.max(all_nsides)
        multiorder_map = mhealpy.HealpixMap(
            grid_value / healpy.nside2pixarea(
                max_nside, degrees=False,
            ),
            uniq_array,
            unit="sr-1"
        )
        column_names = ["PROBDENSITY"]
    return multiorder_map, column_names
