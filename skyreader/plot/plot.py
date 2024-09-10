"""For encapsulating the results of an event scan in a single instance."""

# fmt: off
# pylint: skip-file

import logging
import pickle
from pathlib import Path
from typing import List

import healpy  # type: ignore[import]
import matplotlib  # type: ignore[import]
import meander  # type: ignore[import]
import numpy as np
from astropy.io import ascii  # type: ignore[import]
from matplotlib import patheffects
from matplotlib import pyplot as plt
from matplotlib import text
from matplotlib.projections import projection_registry  # type: ignore[import]
import copy

from .plotting_tools import (
    AstroMollweideAxes,
    DecFormatter,
    format_fits_header,
    hp_ticklabels,
    plot_catalog,
    get_space_angles,
)
from .areas import calculate_area, get_contour_areas

from ..result import SkyScanResult

LOGGER = logging.getLogger("skyreader.plot")


# Function to generate ts maps with a gaussian shape,
# x is the angular distance in degrees
def log_gauss(x, sigma):
    return (x/sigma)**2


# Survival function of the rayleigh distribution
def rayleigh_the_survivor(x, sigma):
    return np.exp(-0.5*(x/sigma)**2)


# Survival function of the rayleigh distribution
def rayleigh(x, sigma):
    return (1./(2*np.pi*sigma**2))*np.exp(-0.5*(x/sigma)**2)


class SkyScanPlotter:
    PLOT_SIZE_Y_IN: float = 3.85
    PLOT_SIZE_X_IN: float = 6
    PLOT_DPI_STANDARD = 150
    PLOT_DPI_ZOOMED = 1200
    PLOT_COLORMAP = matplotlib.colormaps['plasma']
    SIGMA_TO_CONTOUR90 = 2.146
    NEUTRINOFLOOR_SIGMA = 0.2

    def __init__(self, output_dir: Path = Path(".")):
        # Set here plotting parameters and things that
        # do not depend on the individual scan.
        self.output_dir = output_dir
        projection_registry.register(AstroMollweideAxes)

    def create_plot(self, result: SkyScanResult, dozoom: bool = False) -> None:
        """Creates a full-sky plot using a meshgrid at fixed resolution.
        Optionally creates a zoomed-in plot. Resolutions are defined in
        PLOT_DPI_STANDARD and PLOT_DPI_ZOOMED. Zoomed mode is very inefficient
        as the meshgrid is created for the full sky.
        """
        dpi = self.PLOT_DPI_STANDARD if not dozoom else self.PLOT_DPI_ZOOMED

        # number of  grid points along RA coordinate
        xsize = int(self.PLOT_SIZE_X_IN * dpi)
        ysize = int(xsize // 2)  # number of grid points along dec coordinate
        dec = np.linspace(-np.pi/2., np.pi/2., ysize)
        ra = np.linspace(0., 2.*np.pi, xsize)
        # project the map to a rectangular matrix xsize x ysize
        RA, DEC = np.meshgrid(ra, dec)

        # We may want to recover plotting in zenith and azimuth in the future.
        # theta = np.linspace(np.pi, 0., ysize)
        # phi   = np.linspace(0., 2.*np.pi, xsize)

        nsides = result.nsides
        LOGGER.info(f"available nsides: {nsides}")

        event_metadata = result.get_event_metadata()
        unique_id = f'{str(event_metadata)}_{result.get_nside_string()}'
        run_str = f"Run: {event_metadata.run_id}"
        evt_str = f"Event: {event_metadata.event_id}"
        typ_str = f"Type: {event_metadata.event_type}"
        mjd_str = f"MJD: {event_metadata.mjd}"
        plot_title = f"{run_str} {evt_str} {typ_str} {mjd_str}"

        if dozoom:
            addition_to_filename = 'plot_zoomed_legacy.'
        else:
            addition_to_filename = ''
        plot_filename = f"{unique_id}.{addition_to_filename}pdf"
        LOGGER.info(f"saving plot to {plot_filename}")

        min_llh, max_llh = np.nan, np.nan
        min_ra, min_dec = 0., 0.
        grid_map, grid_pix = None, None

        # now plot maps above each other
        for nside in nsides:
            LOGGER.info(f"constructing map for nside {nside}...")
            # grid_pix = healpy.ang2pix(nside, THETA, PHI)
            grid_pix = healpy.ang2pix(nside, np.pi/2. - DEC, RA)
            this_map = np.ones(healpy.nside2npix(nside))*np.inf

            for pixel_data in result.result[f'nside-{nside}']:
                pixel = pixel_data['index']
                # show 2*delta_LLH
                value = 2*pixel_data['llh']
                if np.isfinite(value):
                    if np.isnan(min_llh) or value < min_llh:
                        minCoDec, min_ra = healpy.pix2ang(nside, pixel)
                        min_dec = np.pi/2 - minCoDec
                        min_llh = value
                    if np.isnan(max_llh) or value > max_llh:
                        max_llh = value
                this_map[pixel] = value

            if grid_map is None:
                grid_map = this_map[grid_pix]
            else:
                grid_map = np.where(
                    np.isfinite(this_map[grid_pix]),
                    this_map[grid_pix],
                    grid_map
                )

            del this_map

            LOGGER.info(f"Completed map for nside {nside}.")

        # clean up
        if grid_pix is not None:
            del grid_pix

        if grid_map is None:
            # create an "empty" map if there are no pixels at all
            grid_pix = healpy.ang2pix(8, np.pi/2 - DEC, RA)
            this_map = np.ones(healpy.nside2npix(8))*np.inf
            grid_map = this_map[grid_pix]
            del this_map
            del grid_pix

        LOGGER.info(
            f"min  RA: {min_ra * 180./np.pi} deg, {min_ra*12./np.pi} hours."
        )
        LOGGER.info(f"min Dec: {min_dec * 180./np.pi} deg")

        # renormalize
        if dozoom:
            grid_map = grid_map - min_llh
            min_llh = 0.
            max_llh = 50
        prob_map = copy.copy(grid_map) - min_llh
        prob_map = np.exp(-prob_map)
        min_prob = 1e-64
        max_prob = np.nanmax(prob_map)
        prob_map = prob_map.clip(min_prob, None).astype('float32')


        grid_map = np.ma.masked_invalid(grid_map)

        LOGGER.info(f"Preparing plot: {plot_filename}...")

        # the color map to use
        cmap = self.PLOT_COLORMAP
        cmap.set_under(alpha=0.)  # make underflows transparent
        cmap.set_bad(alpha=1., color=(1., 0., 0.))  # make NaNs bright red

        # prepare the figure canvas
        fig = matplotlib.pyplot.figure(
            figsize=(self.PLOT_SIZE_X_IN, self.PLOT_SIZE_Y_IN)
        )

        ax = None

        if dozoom:
            ax = fig.add_subplot(111)  # ,projection='cartesian')
        else:
            cmap.set_over(alpha=0.)  # make underflows transparent
            ax = fig.add_subplot(111, projection='astro mollweide')

        # rasterized makes the map bitmap while the labels remain vectorial
        # flip longitude to the astro convention
        image = ax.pcolormesh(
            ra,
            dec,
            prob_map,
            #grid_map,
            vmin=min_prob,
            vmax=max_prob,
            norm="log",
            #vmin=min_llh,
            #vmax=max_llh,
            rasterized=True,
            cmap=cmap
        )
        # ax.set_xlim(np.pi, -np.pi)

        contour_levels = (np.array([1.39, 4.61, 11.83, 28.74])+min_llh)[:2]
        contour_labels = [r'50%', r'90%', r'3$\sigma$', r'5$\sigma$'][:2]
        contour_colors = ['k', 'r', 'g', 'b'][:2]
        leg_element = []
        cs_collections = []
        for level, color in zip(contour_levels, contour_colors):
            contour_set = ax.contour(
                ra, dec, grid_map, levels=[level], colors=[color]
            )
            cs_collections.append(contour_set.collections[0])
            e, _ = contour_set.legend_elements()
            leg_element.append(e[0])

        if not dozoom:
            # graticule
            if isinstance(ax, AstroMollweideAxes):
                # mypy guard
                ax.set_longitude_grid(30)
                ax.set_latitude_grid(30)
            cb = fig.colorbar(
                image,
                orientation='horizontal',
                shrink=.6,
                pad=0.05,
                #ticks=[min_llh, max_llh]
                ticks=[min_prob, max_prob],
                norm="log",
            )
            cb.ax.xaxis.set_label_text(r"$-2 \ln(L)$")
        else:
            ax.set_xlabel('right ascension')
            ax.set_ylabel('declination')
            cb = fig.colorbar(
                image, orientation='horizontal', shrink=.6, pad=0.13
            )
            cb.ax.xaxis.set_label_text(r"$-2 \Delta \ln (L)$")

            leg_labels = []
            for i in range(len(contour_labels)):
                vs = cs_collections[i].get_paths()[0].vertices
                # Compute area enclosed by vertices.
                # Take absolute values to be independent of orientation of
                # the boundary integral.
                contour_area = abs(calculate_area(vs))  # in square-radians
                # convert to square-degrees
                contour_area_sqdeg = contour_area*(180.*180.)/(np.pi*np.pi)

                area_string = f"area: {contour_area_sqdeg:.2f}sqdeg"
                leg_labels.append(
                    f'{contour_labels[i]} - {area_string}'
                )

            ax.scatter(
                min_ra,
                min_dec,
                s=20,
                marker='*',
                color='black',
                label=r'scan best-fit',
                zorder=2
            )
            ax.legend(
                leg_element,
                leg_labels,
                loc='lower right',
                fontsize=8,
                scatterpoints=1,
                ncol=2
            )

            LOGGER.info(f"Contour Area (90%): {contour_area_sqdeg} "
                        f"degrees (cartesian) "
                        f"{contour_area_sqdeg * np.cos(min_dec)**2} "
                        "degrees (scaled)")
            x_width = 1.6 * np.sqrt(contour_area_sqdeg)
            LOGGER.info(f"x width is {x_width}")
            if np.isnan(x_width):
                # this get called only when contour_area / x_width is
                # NaN so possibly never invoked in typical situations
                raise RuntimeError(
                    "Estimated area / width is NaN and the fallback logic "
                    "for this scenario is no longer supported. If you "
                    "encounter this error raise an issue to SkyReader."
                )
                # mypy error: "QuadContourSet" has no attribute "allsegs"
                # [attr-defined]. This attribute is likely deprecated but
                # this scenario is rarely (if ever) hit original code is
                # kept commented for the time being

                # note: contour_set is re-assigned at every iteration of
                # the loop on contour_levels, contour_colors, so this
                # effectively corresponds to the last contour_set
                # x_width = 1.6*(max(contour_set.allsegs[i][0][:,0]) -
                # min(contour_set.allsegs[i][0][:,0]))

            y_width = 0.5 * x_width

            lower_x = max(min_ra - x_width*np.pi/180., 0.)
            upper_x = min(min_ra + x_width*np.pi/180., 2 * np.pi)
            lower_y = max(min_dec - y_width*np.pi/180., -np.pi/2.)
            upper_y = min(min_dec + y_width*np.pi/180., np.pi/2.)

            ax.set_xlim(upper_x, lower_x)
            ax.set_ylim(lower_y, upper_y)

            # why not RAFormatter?
            ax.xaxis.set_major_formatter(DecFormatter())

            ax.yaxis.set_major_formatter(DecFormatter())

            factor = 0.25*(np.pi/180.)
            while (upper_x - lower_x)/factor > 6:
                factor *= 2.
            tick_label_grid = factor

            ax.xaxis.set_major_locator(
                matplotlib.ticker.MultipleLocator(base=tick_label_grid)
            )
            ax.yaxis.set_major_locator(
                matplotlib.ticker.MultipleLocator(base=tick_label_grid)
            )

        # cb.ax.xaxis.labelpad = -8
        # workaround for issue with viewers, see colorbar docstring
        # mypy compliance: since cb.solids could be None, we check that
        # it is actually a valid object before accessing it
        if isinstance(cb.solids, matplotlib.collections.QuadMesh):
            cb.solids.set_edgecolor("face")

        if dozoom:
            ax.set_aspect('equal')

        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)

        # show the grid
        ax.grid(True, color='k', alpha=0.5)

        # Otherwise, add the path effects.
        # mypy requires set_path_effects() to take a list of AbstractPathEffect
        effects: List[patheffects.AbstractPathEffect] = [
            patheffects.withStroke(linewidth=1.1, foreground='w')
        ]
        for artist in ax.findobj(text.Text):
            # mypy error: Argument 1 to "set_path_effects" of "Artist"
            # has incompatible type "list[withStroke]"; expected
            # "list[AbstractPathEffect]"  [arg-type]
            artist.set_path_effects(effects)

        # remove white space around figure
        spacing = 0.01
        if not dozoom:
            fig.subplots_adjust(
                bottom=spacing,
                top=1.-spacing,
                left=spacing+0.04,
                right=1.-spacing
            )
        else:
            fig.subplots_adjust(
                bottom=spacing,
                top=0.92-spacing,
                left=spacing+0.1,
                right=1.-spacing
            )

        # set the title
        fig.suptitle(plot_title)

        LOGGER.info(f"saving: {plot_filename}...")

        fig.savefig(self.output_dir / plot_filename, dpi=dpi, transparent=True)

        LOGGER.info("done.")

    @staticmethod
    def circular_contour(ra, dec, sigma, nside):
        """For plotting circular contours on skymaps ra, dec, sigma all
        expected in radians."""
        dec = np.pi/2. - dec
        sigma = np.rad2deg(sigma)
        delta, step, bins = 0, 0, 0
        delta = sigma/180.0*np.pi
        step = 1./np.sin(delta)/10.
        bins = int(360./step)
        Theta = np.zeros(bins+1, dtype=np.double)
        Phi = np.zeros(bins+1, dtype=np.double)
        # define the contour
        for j in range(0, bins):
            phi = j*step/180.*np.pi
            comp1 = np.cos(phi)*np.sin(ra)*np.sin(delta)
            comp2 = np.cos(ra)*np.cos(delta)*np.sin(dec)
            comp3 = np.cos(dec)*np.sin(delta)*np.sin(phi)
            comp4 = np.cos(delta)*np.sin(dec)*np.sin(ra)
            comp5 = np.sin(delta)*np.cos(ra)*np.cos(phi)
            comp6 = np.sin(delta)*np.cos(dec)*np.sin(ra)*np.sin(phi)
            comp7 = np.cos(dec)*np.cos(delta)
            comp8 = np.sin(dec)*np.sin(delta)*np.sin(phi)
            vx = comp1 + comp2 + comp3
            vy = comp4 - comp5 + comp6
            vz = comp7 - comp8
            idx = healpy.vec2pix(nside, vx, vy, vz)
            DEC, RA = healpy.pix2ang(nside, idx)
            Theta[j] = DEC
            Phi[j] = RA
        Theta[bins] = Theta[0]
        Phi[bins] = Phi[0]
        return Theta, Phi

    def create_plot_zoomed(self,
                           result: SkyScanResult,
                           extra_ra=np.nan,
                           extra_dec=np.nan,
                           extra_radius=np.nan,
                           systematics=False,
                           plot_bounding_box=False,
                           plot_4fgl=False,
                           circular=False,
                           circular_err50=0.2,
                           circular_err90=0.7,
                           circularized_ts_map=False,
                           neutrino_floor=False):
        """Uses healpy to plot a map."""

        def bounding_box(ra, dec, theta, phi):
            shift = ra-180

            ra_plus = np.max((np.degrees(phi)-shift) % 360) - 180
            ra_minus = np.min((np.degrees(phi)-shift) % 360) - 180
            dec_plus = (np.pi/2-np.min(theta))*180./np.pi - dec
            dec_minus = (np.pi/2-np.max(theta))*180./np.pi - dec
            return ra_plus, ra_minus, dec_plus, dec_minus

        dpi = self.PLOT_DPI_ZOOMED

        lonra = [-10., 10.]
        latra = [-10., 10.]

        event_metadata = result.get_event_metadata()
        unique_id = f'{str(event_metadata)}_{result.get_nside_string()}'
        run_str = f"Run: {event_metadata.run_id}"
        evt_str = f"Event: {event_metadata.event_id}"
        typ_str = f"Type: {event_metadata.event_type}"
        mjd_str = f"MJD: {event_metadata.mjd}"
        plot_title = f"{run_str} {evt_str} {typ_str} {mjd_str}"

        nsides = result.nsides
        LOGGER.info(f"available nsides: {nsides}")

        if systematics is not True:
            plot_filename = unique_id + ".plot_zoomed_wilks.pdf"
        else:
            plot_filename = unique_id + ".plot_zoomed.pdf"

        nsides = result.nsides
        LOGGER.info(f"available nsides: {nsides}")

        grid_map = dict()
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
            LOGGER.info(f"done with map for nside {nside}...")

        grid_dec_list, grid_ra_list, grid_value_list = [], [], []

        for (dec, ra), value in grid_map.items():
            grid_dec_list.append(dec)
            grid_ra_list.append(ra)
            grid_value_list.append(value)
        grid_dec: np.ndarray = np.asarray(grid_dec_list)
        grid_ra: np.ndarray = np.asarray(grid_ra_list)
        grid_value: np.ndarray = np.asarray(grid_value_list)

        sorting_indices = np.argsort(grid_value)
        grid_value = grid_value[sorting_indices]
        grid_dec = grid_dec[sorting_indices]
        grid_ra = grid_ra[sorting_indices]

        min_value = grid_value[0]
        min_dec = grid_dec[0]
        min_ra = grid_ra[0]

        LOGGER.info(
            f"min  RA: {min_ra * 180./np.pi} deg, {min_ra*12./np.pi} hours."
        )
        LOGGER.info(f"min Dec: {min_dec * 180./np.pi} deg")

        # renormalize
        grid_value = grid_value - min_value
        min_value = 0.

        # renormalize
        equatorial_map[np.isinf(equatorial_map)] = np.nan
        equatorial_map -= np.nanmin(equatorial_map)

        # Convert to probability
        equatorial_map = np.exp(-1. * equatorial_map)
        equatorial_map = equatorial_map / np.nansum(equatorial_map)

        # nan values are a problem for the convolution
        equatorial_map[np.isnan(equatorial_map)] = 0.

        if neutrino_floor:
            # convolute with a gaussian with 0.2 deg as sigma
            equatorial_map = healpy.smoothing(
                equatorial_map,
                sigma=np.deg2rad(self.NEUTRINOFLOOR_SIGMA),
            )

            # normalize map
            equatorial_map[np.isnan(equatorial_map)] = 0.
            equatorial_map = equatorial_map.clip(0., None)
            normalization = np.nansum(equatorial_map)
            equatorial_map = equatorial_map / normalization

        # avoid excessively heavy data format
        equatorial_map = equatorial_map.clip(1.e-12, None).astype('float32')

        # obtain values for grid map
        grid_value = healpy.get_interp_val(
            equatorial_map, np.pi/2 - grid_dec, grid_ra
        )
        sorted_values = np.sort(equatorial_map)[::-1]

        # Calculate the contours
        if systematics:
            # from Pan-Starrs event 127852
            # these are values determined from MC by Will on the TS (2*LLH)
            # Not clear yet how to translate this for the probability map
            contour_levels = (np.array([22.2, 64.2])+min_value)
            contour_labels = [
                r'50% (IC160427A syst.)', r'90% (IC160427A syst.)'
            ]
            contour_colors = ['k', 'r']
        else:
            # Wilks
            probability_levels = (
                np.array([0.5, 0.9, 1-1.35e-3, 1-2.87e-7])
            )[:3]
            contour_labels = [r'50%', r'90%', r'3$\sigma$', r'5$\sigma$'][:3]
            contour_colors = ['k', 'r', 'g', 'b'][:3]

        sample_points = np.array([np.pi/2 - grid_dec, grid_ra]).T

        contour_levels = list()
        for prob in probability_levels:
            level_index = (
                np.nancumsum(sorted_values) >= prob
            ).tolist().index(True)
            level = (
                sorted_values[level_index] + (
                    sorted_values[level_index+1] if level_index+1<len(
                        sorted_values
                    ) else 0
                )
            )/2.0
            contour_levels.append(level)

        # Get contours from healpix map
        contours_by_level = meander.spherical_contours(
            sample_points,
            np.log(grid_value),
            np.log(contour_levels),
        )

        # Calculate areas using Gauss-Green's theorem for a spherical space
        contour_areas = get_contour_areas(contours_by_level, min_ra)

        # In case it is requested, check if neutrino floor must be applied
        if neutrino_floor:
            neutrino_floor_90_area = np.pi * (
                self.NEUTRINOFLOOR_SIGMA * self.SIGMA_TO_CONTOUR90)**2
            if circularized_ts_map:
                compare_90_area = np.pi * circular_err90**2
            else:
                compare_90_area = contour_areas[1]
            if compare_90_area < neutrino_floor_90_area:
                circularized_ts_map = True
                circular_err90 = (
                    self.NEUTRINOFLOOR_SIGMA * self.SIGMA_TO_CONTOUR90
                )
                LOGGER.info("Contour too small, applying neutrino floor...")

        if circularized_ts_map:
            main_plot_filename = plot_filename.split(".pdf")[-2]
            plot_filename = main_plot_filename + "_circularized_ts.pdf"

        LOGGER.info(f"saving plot to {plot_filename}")

        # In case it is requested, generate a mock ts map with a gaussian shape
        # around the best fit direction
        if circularized_ts_map:

            LOGGER.info("Generating a circularized ts map...")

            min_index = np.nanargmin(equatorial_map)

            space_angle, ang_dist_grid = get_space_angles(
                min_ra, min_dec, grid_ra, grid_dec, max_nside, min_index
            )

            event_sigma = circular_err90/self.SIGMA_TO_CONTOUR90
            equatorial_map = log_gauss(space_angle, event_sigma)
            grid_value = log_gauss(ang_dist_grid, event_sigma)

            # re-calculate areas
            contours_by_level = meander.spherical_contours(
                sample_points,
                grid_value,
                contour_levels
            )
            contour_areas = get_contour_areas(contours_by_level, min_ra)

        LOGGER.info(f"preparing plot: {plot_filename}...")

        cmap = self.PLOT_COLORMAP
        cmap.set_under('w')
        cmap.set_bad(alpha=1., color=(1., 0., 0.))  # make NaNs bright red

        # In case it is desired, just draw circular contours over the ts map
        if circular:
            sigma50 = np.deg2rad(circular_err50)
            sigma90 = np.deg2rad(circular_err90)
            Theta50, Phi50 = self.circular_contour(
                min_ra, min_dec, sigma50, nside
            )
            Theta90, Phi90 = self.circular_contour(
                min_ra, min_dec, sigma90, nside
            )
            contour50 = np.dstack((Theta50, Phi50))
            contour90 = np.dstack((Theta90, Phi90))
            contours_by_level = [contour50, contour90]

            # re-calculate areas
            contour_areas = get_contour_areas(contours_by_level, min_ra)

        # Check for RA values that are out of bounds
        for level in contours_by_level:
            for contour in level:
                contour.T[1] = np.where(
                    contour.T[1] < 0.,
                    contour.T[1] + 2.*np.pi,
                    contour.T[1]
                )

        # Find the rough extent of the contours to bound the plot
        contours = contours_by_level[-1]
        ra = min_ra * 180./np.pi
        dec = min_dec * 180./np.pi
        theta, phi = np.concatenate(contours_by_level[-1]).T
        ra_plus, ra_minus, dec_plus, dec_minus = bounding_box(
            ra, dec, theta, phi
        )
        ra_bound = min(15, max(3, max(-ra_minus, ra_plus)))
        dec_bound = min(15, max(2, max(-dec_minus, dec_plus)))
        lonra = [-ra_bound, ra_bound]
        latra = [-dec_bound, dec_bound]

        # Begin the figure
        plt.clf()
        # Rotate into healpy coordinates
        lon, lat = np.degrees(min_ra), np.degrees(min_dec)
        max_prob = np.nanmax(equatorial_map)
        if len(contour_levels) >= 3:
            min_prob = contour_levels[2]
        else:
            min_prob = max_prob/1e8
        healpy.cartview(
            map=equatorial_map,
            title=plot_title,
            min=min_prob,  # min 2DeltaLLH value for colorscale
            max=max_prob,  # max 2DeltaLLH value for colorscale
            rot=(lon, lat, 0.),
            cmap=cmap,
            hold=True,
            cbar=None,
            lonra=lonra,
            latra=latra,
            norm='log',
            unit="Probability",
        )

        fig = plt.gcf()
        ax = plt.gca()
        image = ax.get_images()[0]
        # Place colorbar by hand
        cb = fig.colorbar(
            image,
            ax=ax,
            orientation='horizontal',
            aspect=50,
            ticks = [
                min_prob,
                min_prob*(max_prob/min_prob)**(1/5),
                min_prob*(max_prob/min_prob)**(2/5),
                min_prob*(max_prob/min_prob)**(3/5),
                min_prob*(max_prob/min_prob)**(4/5),
                max_prob
            ],
            format="{x:.1e}"
        )
        cb.ax.xaxis.set_label_text("Probability")

        # Plot the best-fit location
        # This requires some more coordinate transformations
        healpy.projplot(
            np.pi/2 - min_dec,
            min_ra,
            '*',
            ms=5,
            label=r'scan best fit',
            color='black',
            zorder=2
        )

        # Plot the contours
        for (
            contour_area_sqdeg,
            contour_label,
            contour_color,
            contours
        ) in zip(
            contour_areas,
            contour_labels,
            contour_colors,
            contours_by_level
        ):
            contour_label = contour_label + ' - area: {0:.2f} sqdeg'.format(
                contour_area_sqdeg)
            first = True
            for contour in contours:
                theta, phi = contour.T
                if first:
                    healpy.projplot(
                        theta,
                        phi,
                        linewidth=2,
                        c=contour_color,
                        label=contour_label
                    )
                else:
                    healpy.projplot(theta, phi, linewidth=2, c=contour_color)
                first = False

        # Add some grid lines
        healpy.graticule(dpar=2, dmer=2, force=True)

        # Set some axis limits
        lower_ra = min_ra + np.radians(lonra[0])
        upper_ra = min_ra + np.radians(lonra[1])
        lower_dec = min_dec + np.radians(latra[0])
        upper_dec = min_dec + np.radians(latra[1])

        lower_lon = np.degrees(lower_ra)
        upper_lon = np.degrees(upper_ra)
        tmp_lower_lat = np.degrees(lower_dec)
        tmp_upper_lat = np.degrees(upper_dec)
        lower_lat = min(tmp_lower_lat, tmp_upper_lat)
        upper_lat = max(tmp_lower_lat, tmp_upper_lat)

        # Label the axes
        hp_ticklabels(
            zoom=True,
            lonra=lonra,
            latra=latra,
            rot=(lon, lat, 0),
            bounds=(lower_lon, upper_lon, lower_lat, upper_lat)
        )

        if plot_4fgl:
            # Overlay 4FGL sources
            plot_catalog(
                equatorial_map, cmap, lower_ra, upper_ra, lower_dec, upper_dec
            )

        # Approximate contours as rectangles
        ra = min_ra * 180./np.pi
        dec = min_dec * 180./np.pi
        percentages = ["50", "90"]
        for l_index, contours in enumerate(contours_by_level[:2]):
            ra_plus = None
            theta, phi = np.concatenate(contours).T
            ra_plus, ra_minus, dec_plus, dec_minus = bounding_box(
                ra, dec, theta, phi
            )
            contain_txt = "Approximating the " + percentages[l_index] + \
                "% error region as a rectangle, we get:" + " \n" + \
                          "\t RA = {0:.2f} + {1:.2f} - {2:.2f}".format(
                              ra, ra_plus, np.abs(ra_minus)) + " \n" + \
                          "\t Dec = {0:.2f} + {1:.2f} - {2:.2f}".format(
                              dec, dec_plus, np.abs(dec_minus))
            # This is actually an output and not a logging info.
            # TODO: we should wrap this in an object, return and log at
            # the higher level.
            print(contain_txt)

        print("Contour Area (50%):", contour_areas[0], "square degrees (scaled)")
        print("Contour Area (90%):", contour_areas[1], "square degrees (scaled)")

        if plot_bounding_box:
            bounding_ras_list, bounding_decs_list = [], []
            # lower bound
            bounding_ras_list.extend(list(np.linspace(
                ra+ra_minus,
                ra+ra_plus,
                10
            )))
            bounding_decs_list.extend([dec+dec_minus]*10)
            # right bound
            bounding_ras_list.extend([ra+ra_plus]*10)
            bounding_decs_list.extend(list(np.linspace(
                dec+dec_minus,
                dec+dec_plus,
                10
            )))
            # upper bound
            bounding_ras_list.extend(list(np.linspace(
                ra+ra_plus,
                ra+ra_minus,
                10
            )))
            bounding_decs_list.extend([dec+dec_plus]*10)
            # left bound
            bounding_ras_list.extend([ra+ra_minus]*10)
            bounding_decs_list.extend(list(np.linspace(
                dec+dec_plus,
                dec+dec_minus,
                10
            )))
            # join end to beginning
            bounding_ras_list.append(bounding_ras_list[0])
            bounding_decs_list.append(bounding_decs_list[0])

            bounding_ras: np.ndarray = np.asarray(bounding_ras_list)
            bounding_decs: np.ndarray = np.asarray(bounding_decs_list)
            bounding_phi = np.radians(bounding_ras)
            bounding_theta = np.pi/2 - np.radians(bounding_decs)
            bounding_contour = np.array([bounding_theta, bounding_phi])
            bounding_contour_area = 0.
            bounding_contour_area = abs(calculate_area(bounding_contour.T))
            # convert to square-degrees
            bounding_contour_area *= (180.*180.)/(np.pi*np.pi)
            contour_label = r'90% Bounding rectangle' + \
                ' - area: {0:.2f} sqdeg'.format(bounding_contour_area)
            healpy.projplot(
                bounding_theta,
                bounding_phi,
                linewidth=0.75,
                c='r',
                linestyle='dashed',
                label=contour_label
            )

        # Output contours in RA, dec instead of theta, phi
        saving_contours: list = []
        for contours in contours_by_level:
            saving_contours.append([])
            for contour in contours:
                saving_contours[-1].append([])
                theta, phi = contour.T
                ras = phi
                decs = np.pi/2 - theta
                for tmp_ra, tmp_dec in zip(ras, decs):
                    saving_contours[-1][-1].append([tmp_ra, tmp_dec])

        # Save the individual contours, send messages
        for i, val in enumerate(["50", "90"]):
            ras = list(np.asarray(saving_contours[i][0]).T[0])
            decs = list(np.asarray(saving_contours[i][0]).T[1])
            tab = {"ra (rad)": ras, "dec (rad)": decs}
            savename = unique_id + ".contour_" + val + ".txt"
            try:
                LOGGER.info("Dumping to {savename}")
                ascii.write(tab, savename, overwrite=True)
            except OSError:
                LOGGER.error(
                    "OS Error prevented contours from being written, "
                    "maybe a memory issue. Error is:\n{err}")

        uncertainty = [(ra_minus, ra_plus), (dec_minus, dec_plus)]
        fits_header = format_fits_header(
            (
                event_metadata.run_id,
                event_metadata.event_id,
                event_metadata.event_type
            ),
            0,
            np.degrees(min_ra),
            np.degrees(min_dec),
            uncertainty,
        )
        mmap_nside = healpy.get_nside(equatorial_map)

        # Plot the original online reconstruction location
        if np.sum(np.isnan([extra_ra, extra_dec, extra_radius])) == 0:

            # dist = angular_distance(minRA, minDec, extra_ra * np.pi/180.,
            # extra_dec * np.pi/180.)
            # print("Millipede best fit is", dist /(np.pi *
            # extra_radius/(1.177 * 180.)), "sigma from reported best fit")

            extra_ra_rad = np.radians(extra_ra)
            extra_dec_rad = np.radians(extra_dec)
            extra_radius_rad = np.radians(extra_radius)
            extra_lon = extra_ra_rad
            extra_lat = extra_dec_rad

            healpy.projscatter(
                np.degrees(extra_lon),
                np.degrees(extra_lat),
                lonlat=True,
                c='m',
                marker='x',
                s=20,
                label=r'Reported online (50%, 90%)'
            )
            for cont_scale, cont_col, cont_sty in zip(
                    [1., 2.1459/1.177], ['m', 'm'], ['-', '--']
            ):
                spline_contour = self.circular_contour(
                    extra_ra_rad,
                    extra_dec_rad,
                    extra_radius_rad*cont_scale,
                    healpy.get_nside(equatorial_map)
                )
                spline_lon = spline_contour[1]
                spline_lat = np.pi/2. - spline_contour[0]
                healpy.projplot(
                    np.degrees(spline_lon),
                    np.degrees(spline_lat),
                    lonlat=True,
                    linewidth=2.,
                    color=cont_col,
                    linestyle=cont_sty
                )

        plt.legend(fontsize=6, loc="lower left")

        # Dump the whole contour
        path = unique_id + ".contour.pkl"
        print("Saving contour to", path)
        with open(path, "wb") as f:
            pickle.dump(saving_contours, f)

        healpy.write_map(
            f"{unique_id}.skymap_nside_{mmap_nside}.fits.gz",
            equatorial_map,
            coord='C',
            column_names=['2LLH'],
            extra_header=fits_header,
            overwrite=True
        )

        # Save the figure
        LOGGER.info(f"Saving: {plot_filename}...")
        # ax.invert_xaxis()
        fig.savefig(
            self.output_dir / plot_filename,
            dpi=dpi,
            transparent=True,
            bbox_inches='tight'
        )

        LOGGER.info("done.")

        plt.close()
