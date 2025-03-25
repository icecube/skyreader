"""For encapsulating the results of an event scan in a single instance."""

# fmt: off
# pylint: skip-file

import logging
import pickle
from pathlib import Path
from typing import List, Union

import copy
import healpy  # type: ignore[import]
import matplotlib  # type: ignore[import]
import meander  # type: ignore[import]
import numpy as np
from astropy.io import ascii  # type: ignore[import]
from matplotlib import patheffects
from matplotlib import pyplot as plt
from matplotlib import text
from matplotlib.projections import projection_registry  # type: ignore[import]

from .plotting_tools import (
    AstroMollweideAxes,
    format_fits_header,
    hp_ticklabels,
    plot_catalog
)

from ..utils.areas import calculate_area, get_contour_areas
from ..utils.handle_map_data import (
    extract_map,
    get_contour_levels,
    prepare_flattened_map,
    prepare_multiorder_map,
)
from ..result import SkyScanResult

LOGGER = logging.getLogger("skyreader.plot")


class SkyScanPlotter:
    PLOT_SIZE_Y_IN: float = 3.85
    PLOT_SIZE_X_IN: float = 6
    PLOT_DPI_STANDARD = 300
    PLOT_DPI_ZOOMED = 1200
    PLOT_COLORMAP = matplotlib.colormaps['plasma_r']

    def __init__(self, output_dir: Path = Path(".")):
        # Set here plotting parameters and things that
        # do not depend on the individual scan.
        self.output_dir = output_dir
        projection_registry.register(AstroMollweideAxes)

    def create_plot(
        self,
        result: SkyScanResult,
        systematics: bool = False,
        llh_map: bool = True,
        angular_error_floor: Union[None, float] = None,
    ) -> None:
        """Creates a full-sky plot using a meshgrid at fixed resolution.
        Optionally creates a zoomed-in plot. Resolutions are defined in
        PLOT_DPI_STANDARD and PLOT_DPI_ZOOMED. Zoomed mode is very inefficient
        as the meshgrid is created for the full sky.
        """
        dpi = self.PLOT_DPI_STANDARD

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

        plot_filename = f"{unique_id}.pdf"
        LOGGER.info(f"saving plot to {plot_filename}")

        grid_value, grid_ra, grid_dec, equatorial_map, _ = extract_map(
            result,
            llh_map,
            angular_error_floor,
            remove_min_val=not llh_map,
        )

        grid_pix = healpy.ang2pix(max(nsides), np.pi/2. - DEC, RA)
        plotting_map = equatorial_map[grid_pix]

        min_dec = grid_dec[0]
        min_ra = grid_ra[0]

        LOGGER.info(
            f"min  RA: {min_ra * 180./np.pi} deg, {min_ra*12./np.pi} hours."
        )
        LOGGER.info(f"min Dec: {min_dec * 180./np.pi} deg")

        # renormalize
        if llh_map:
            cmap = self.PLOT_COLORMAP
            text_colorbar = r"$-2 \ln(L)$"
            vmin = np.nanmin(equatorial_map)
            vmax = np.nanmax(equatorial_map)
            map_to_plot = plotting_map
        else:
            cmap = matplotlib.colormaps[
                self.PLOT_COLORMAP.name.rstrip('_r')
            ]
            text_colorbar = r"log10$(p)$"
            vmin = np.nanmin(
                np.log10(equatorial_map[equatorial_map != 0.])
            )
            vmax = np.nanmax(
                np.log10(equatorial_map[equatorial_map != 0.])
            )
            map_to_plot = copy.copy(plotting_map)
            map_to_plot[plotting_map != 0.] = np.log10(
                plotting_map[plotting_map != 0.]
            )
            map_to_plot[plotting_map == 0.] = np.nan
        equatorial_map = np.ma.masked_invalid(equatorial_map)
        map_to_plot = np.ma.masked_invalid(map_to_plot)

        LOGGER.info(f"Preparing plot: {plot_filename}...")

        # features of the color map to use
        cmap.set_under(alpha=0.)  # make underflows transparent
        cmap.set_bad(alpha=1., color=(1., 0., 0.))  # make NaNs bright red

        # prepare the figure canvas
        fig = matplotlib.pyplot.figure(
            figsize=(self.PLOT_SIZE_X_IN, self.PLOT_SIZE_Y_IN)
        )

        ax = None

        cmap.set_over(alpha=0.)  # make underflows transparent
        ax = fig.add_subplot(111, projection='astro mollweide')

        # rasterized makes the map bitmap while the labels remain vectorial
        # flip longitude to the astro convention
        image = ax.pcolormesh(
            ra,
            dec,
            map_to_plot,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
            cmap=cmap
        )
        # ax.set_xlim(np.pi, -np.pi)

        (
            contour_levels, contour_labels, contour_colors
        ) = get_contour_levels(equatorial_map, llh_map, systematics)

        for level, color in zip(contour_levels, contour_colors):
            if not llh_map:
                level = np.log10(level)
            ax.contour(
                ra, dec, map_to_plot, levels=[level], colors=[color]
            )

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
            ticks=[vmin, vmax],
        )
        cb.ax.xaxis.set_label_text(text_colorbar)

        # cb.ax.xaxis.labelpad = -8
        # workaround for issue with viewers, see colorbar docstring
        # mypy compliance: since cb.solids could be None, we check that
        # it is actually a valid object before accessing it
        if isinstance(cb.solids, matplotlib.collections.QuadMesh):
            cb.solids.set_edgecolor("face")

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
        fig.subplots_adjust(
            bottom=spacing,
            top=1.-spacing,
            left=spacing+0.04,
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

    def create_plot_zoomed(
        self,
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
        angular_error_floor=None,  # if not None, sigma of the
        # gaussian to convolute the map with in deg.
        llh_map=True
    ):
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
        max_nside = max(nsides)
        LOGGER.info(f"available nsides: {nsides}")

        if systematics is not True:
            plot_filename = unique_id + ".plot_zoomed_wilks.pdf"
        else:
            plot_filename = unique_id + ".plot_zoomed.pdf"

        (
            grid_value, grid_ra, grid_dec, equatorial_map, uniq_array
        ) = extract_map(result, llh_map, angular_error_floor)
        min_dec = grid_dec[0]
        min_ra = grid_ra[0]

        LOGGER.info(
            f"min  RA: {min_ra * 180./np.pi} deg, {min_ra*12./np.pi} hours."
        )
        LOGGER.info(f"min Dec: {min_dec * 180./np.pi} deg")

        (
            contour_levels, contour_labels, contour_colors
        ) = get_contour_levels(equatorial_map, llh_map, systematics)

        sample_points = np.array([np.pi/2 - grid_dec, grid_ra]).T

        if not circular:
            grid_values_for_contours = copy.copy(grid_value)
            if llh_map:
                # get rid of nan values only for the contours
                # this avoids crashes during plotting
                grid_values_for_contours[
                    np.isnan(grid_values_for_contours)
                ] = np.nanmax(equatorial_map)
                contour_levels_for_contours = contour_levels
            else:
                grid_values_for_contours[
                    np.logical_or(
                        np.isnan(grid_values_for_contours),
                        grid_values_for_contours == 0.
                    )
                ] = np.nanmin(equatorial_map[equatorial_map > 0.])
                grid_values_for_contours = np.log(
                    grid_values_for_contours
                )
                contour_levels_for_contours = np.log(contour_levels)
            # Get contours from healpix map
            contours_by_level = meander.spherical_contours(
                sample_points,
                grid_values_for_contours,
                contour_levels_for_contours,
            )

        LOGGER.info(f"saving plot to {plot_filename}")
        LOGGER.info(f"preparing plot: {plot_filename}...")

        # In case it is desired, just draw circular contours over the ts map
        if circular:
            sigma50 = np.deg2rad(circular_err50)
            sigma90 = np.deg2rad(circular_err90)
            Theta50, Phi50 = self.circular_contour(
                min_ra, min_dec, sigma50, max_nside
            )
            Theta90, Phi90 = self.circular_contour(
                min_ra, min_dec, sigma90, max_nside
            )
            contour50 = np.dstack((Theta50, Phi50))
            contour90 = np.dstack((Theta90, Phi90))
            contours_by_level = [contour50, contour90]

        # Calculate areas using Gauss-Green's theorem for a spherical space
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
        if llh_map:
            cmap = self.PLOT_COLORMAP
            cmap.set_under('w')
            # make NaNs bright red
            cmap.set_bad(alpha=1., color=(1., 0., 0.))
            healpy.cartview(
                map=equatorial_map,
                title=plot_title,
                min=0.,  # min 2DeltaLLH value for colorscale
                max=40.,  # max 2DeltaLLH value for colorscale
                rot=(lon, lat, 0.),
                cmap=cmap,
                hold=True,
                cbar=None,
                lonra=lonra,
                latra=latra,
                unit=r"$-2 \Delta \ln (L)$",
            )
            ticks = None
            format = None
            cb_label = r"$-2 \Delta \ln (L)$"
        else:
            cmap = matplotlib.colormaps[
                self.PLOT_COLORMAP.name.rstrip('_r')
            ]
            cmap.set_under('w')
            # make NaNs bright red
            cmap.set_bad(alpha=1., color=(1., 0., 0.))
            max_prob = np.nanmax(equatorial_map)
            if len(contour_levels) >= 3:
                min_prob = contour_levels[2]
            else:
                min_prob = max_prob/1e8
            healpy.cartview(
                map=equatorial_map.clip(1e-12, None),
                title=plot_title,
                min=min_prob,  # min prob value for colorscale
                max=max_prob,  # max prob value for colorscale
                rot=(lon, lat, 0.),
                cmap=cmap,
                hold=True,
                cbar=None,
                lonra=lonra,
                latra=latra,
                norm='log',
                unit="Probability",
            )
            ticks = [
                min_prob,
                min_prob*(max_prob/min_prob)**(1/5),
                min_prob*(max_prob/min_prob)**(2/5),
                min_prob*(max_prob/min_prob)**(3/5),
                min_prob*(max_prob/min_prob)**(4/5),
                max_prob
            ]
            format = "{x:.1e}"
            cb_label = "Probability"

        fig = plt.gcf()
        ax = plt.gca()
        image = ax.get_images()[0]
        # Place colorbar by hand
        cb = fig.colorbar(
            image,
            ax=ax,
            orientation='horizontal',
            aspect=50,
            ticks=ticks,
            format=format
        )
        cb.ax.xaxis.set_label_text(cb_label)

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
            contour_label = contour_label + f' - area: {contour_area_sqdeg:.2f} sqdeg'
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
        rectangular_errors = {}
        percentages = ["50", "90"]
        uncertainties = []
        for l_index, contours in enumerate(contours_by_level[:2]):
            ra_plus = None
            theta, phi = np.concatenate(contours).T
            ra_plus, ra_minus, dec_plus, dec_minus = bounding_box(ra, dec, theta, phi)
            # Save the results in the dictionary for each percentage level
            rectangular_errors[percentages[l_index]] = {
                "ra_plus": ra_plus,
                "ra_minus": ra_minus,
                "dec_plus": dec_plus,
                "dec_minus": dec_minus
            }
            # Optional: Print or log the results if needed
            contain_txt = "Approximating the " + percentages[l_index] + \
                "% error region as a rectangle, we get:" + " \n" + \
                          "\t RA = {0:.2f} + {1:.2f} - {2:.2f}".format(
                              ra, ra_plus, np.abs(ra_minus)) + " \n" + \
                          "\t Dec = {0:.2f} + {1:.2f} - {2:.2f}".format(
                              dec, dec_plus, np.abs(dec_minus))
            print(contain_txt)
            uncertainty = [(ra_minus, ra_plus), (dec_minus, dec_plus)]
            uncertainties.append(uncertainty)
            # This is actually an output and not a logging info.
            # TODO: we should wrap this in an object, return and log at
            # the higher level.
        print(
            f"Contour Area (50%): {contour_areas[0]}",
            "square degrees (scaled)"
        )
        print(
            f"Contour Area (90%): {contour_areas[1]}",
            "square degrees (scaled)"
        )

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
                f' - area: {bounding_contour_area:.2f} sqdeg'
            healpy.projplot(
                bounding_theta,
                bounding_phi,
                linewidth=0.75,
                c='r',
                linestyle='dashed',
                label=contour_label
            )

        fits_header = format_fits_header(
            (
                event_metadata.run_id,
                event_metadata.event_id,
                event_metadata.event_type
            ),
            event_metadata.mjd,
            np.degrees(min_ra),
            np.degrees(min_dec),
            uncertainties,
            contour_areas,
            llh_map,
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

        # save flattened map
        equatorial_map, column_names, column_units = prepare_flattened_map(
            equatorial_map, llh_map
        )
        if llh_map:
            type_map = "llh"
        else:
            type_map = "probability"
        filename_main = f"{unique_id}.skymap_nside_{mmap_nside}_{type_map}"
        healpy.write_map(
            self.output_dir / f"{filename_main}.fits.gz",
            equatorial_map,
            coord='C',
            column_names=column_names,
            column_units=column_units,
            extra_header=fits_header,
            overwrite=True,
        )
        multiorder_map, column_names = prepare_multiorder_map(
            grid_value, uniq_array, llh_map, column_names
        )
        multiorder_map.write_map(
            self.output_dir / f"{filename_main}.multiorder.fits.gz",
            column_names=column_names,
            extra_header=fits_header,
            overwrite=True,
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

        self._save_contours(contours_by_level, unique_id)
        LOGGER.info("done.")
        plt.close()

    def _save_contours(self, contours_by_level, unique_id) -> None:
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

        # Save the individual contours
        for i, val in enumerate(["50", "90"]):
            ras = list(np.asarray(saving_contours[i][0]).T[0])
            decs = list(np.asarray(saving_contours[i][0]).T[1])
            tab = {"ra (rad)": ras, "dec (rad)": decs}
            savepath = self.output_dir / f"{unique_id}.contour_{val}.txt"
            try:
                LOGGER.info(f"Dumping to {savepath}")
                ascii.write(tab, savepath, overwrite=True)
            except OSError:
                LOGGER.error(
                    "OS Error prevented contours from being written, "
                    "maybe a memory issue. Error is:\n{err}")

        # Dump the whole contour
        path = self.output_dir / f"{unique_id}.contour.pkl"
        print("Saving contour to", path)
        with open(path, "wb") as f:
            pickle.dump(saving_contours, f)
