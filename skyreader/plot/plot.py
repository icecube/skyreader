"""For encapsulating the results of an event scan in a single instance."""

# fmt: off
# pylint: skip-file
# flake8: noqa

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple, TypedDict, Union

import healpy  # type: ignore[import]
import matplotlib  # type: ignore[import]
import meander  # type: ignore[import]
import numpy as np
from astropy.io import ascii  # type: ignore[import]
from matplotlib import patheffects
from matplotlib import pyplot as plt
from matplotlib import text
from pathlib import Path

from .plotting_tools import (
    AstroMollweideAxes,
    DecFormatter,
    RaFormatter,
    format_fits_header,
    hp_ticklabels,
    plot_catalog,
)

from ..result import SkyScanResult

LOGGER = logging.getLogger("skyreader.plot")

class SkyScanPlotter:
    PLOT_SIZE_Y_IN: float = 3.85
    PLOT_SIZE_X_IN: float = 6
    PLOT_DPI_STANDARD = 150
    PLOT_DPI_ZOOMED = 1200
    PLOT_COLORMAP = matplotlib.colormaps['plasma_r']

    def __init__(self, output_dir: Path = Path(".")):
        # Set here plotting parameters and things that do not depend on the individual scan.
        self.output_dir = output_dir

    @staticmethod
    # Calculates are using Gauss-Green theorem / shoelace formula
    # TODO: vectorize using numpy.
    # Note: in some cases the argument is not a np.ndarray so one has to convert the data series beforehand.
    def calculate_area(vs) -> float:
        a = 0
        x0, y0 = vs[0]
        for [x1,y1] in vs[1:]:
            dx = np.cos(x1)-np.cos(x0)
            dy = y1-y0
            a += 0.5*(y0*dx - np.cos(x0)*dy)
            x0 = x1
            y0 = y1
        return a

    def create_plot(self, result: SkyScanResult, dozoom: bool = False) -> None:
        """Creates a full-sky plot using a meshgrid at fixed resolution.
        Optionally creates a zoomed-in plot. Resolutions are defined in
        PLOT_DPI_STANDARD and PLOT_DPI_ZOOMED. Zoomed mode is very inefficient
        as the meshgrid is created for the full sky.
        """
        dpi = self.PLOT_DPI_STANDARD if not dozoom else self.PLOT_DPI_ZOOMED

        xsize = int(self.PLOT_SIZE_X_IN * dpi) # number of  grid points along RA coordinate
        ysize = int(xsize // 2) # number of grid points along dec coordinate
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
        plot_title = f"Run: {event_metadata.run_id} Event {event_metadata.event_id}: Type: {event_metadata.event_type} MJD: {event_metadata.mjd}"

        plot_filename = f"{unique_id}.{'plot_zoomed_legacy.' if dozoom else ''}pdf"
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
                grid_map = np.where( np.isfinite(this_map[grid_pix]), this_map[grid_pix], grid_map)

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

        LOGGER.info(f"min  RA: {min_ra *180./np.pi} deg, {min_ra*12./np.pi} hours.")
        LOGGER.info(f"min Dec: {min_dec * 180./np.pi} deg")

        # renormalize
        if dozoom:
            grid_map = grid_map - min_llh
            # max_value = max_value - min_value
            min_llh = 0.
            max_llh = 50

        grid_map = np.ma.masked_invalid(grid_map)

        LOGGER.info(f"Preparing plot: {plot_filename}...")

        # the color map to use
        cmap = self.PLOT_COLORMAP
        cmap.set_under(alpha=0.) # make underflows transparent
        cmap.set_bad(alpha=1., color=(1.,0.,0.)) # make NaNs bright red

        # prepare the figure canvas
        fig = matplotlib.pyplot.figure(figsize=(self.PLOT_SIZE_X_IN,self.PLOT_SIZE_Y_IN))

        ax = None

        if dozoom:
            ax = fig.add_subplot(111) #,projection='cartesian')
        else:
            cmap.set_over(alpha=0.)  # make underflows transparent
            ax = fig.add_subplot(111,projection='astro mollweide')

        # rasterized makes the map bitmap while the labels remain vectorial
        # flip longitude to the astro convention
        image = ax.pcolormesh(ra, dec, grid_map, vmin=min_llh, vmax=max_llh, rasterized=True, cmap=cmap)
        # ax.set_xlim(np.pi, -np.pi)



        contour_levels = (np.array([1.39, 4.61, 11.83, 28.74])+min_llh)[:2]
        contour_labels = [r'50%', r'90%', r'3$\sigma$', r'5$\sigma$'][:2]
        contour_colors=['k', 'r', 'g', 'b'][:2]
        leg_element=[]
        cs_collections = []
        for level, color in zip(contour_levels, contour_colors):
            contour_set = ax.contour(ra, dec, grid_map, levels=[level], colors=[color])
            cs_collections.append(contour_set.collections[0])
            e, _ = contour_set.legend_elements()
            leg_element.append(e[0])

        if not dozoom:
            # graticule
            if isinstance(ax, AstroMollweideAxes):
                # mypy guard
                ax.set_longitude_grid(30)
                ax.set_latitude_grid(30)
            cb = fig.colorbar(image, orientation='horizontal', shrink=.6, pad=0.05, ticks=[min_llh, max_llh])
            cb.ax.xaxis.set_label_text(r"$-2 \ln(L)$")
        else:
            ax.set_xlabel('right ascension')
            ax.set_ylabel('declination')
            cb = fig.colorbar(image, orientation='horizontal', shrink=.6, pad=0.13)
            cb.ax.xaxis.set_label_text(r"$-2 \Delta \ln (L)$")

            leg_labels = []
            for i in range(len(contour_labels)):
                vs = cs_collections[i].get_paths()[0].vertices
                # Compute area enclosed by vertices.
                # Take absolute values to be independent of orientation of the boundary integral.
                contour_area = abs(self.calculate_area(vs)) # will be in square-radians
                contour_area_sqdeg = contour_area*(180.*180.)/(np.pi*np.pi) # convert to square-degrees

                leg_labels.append(f'{contour_labels[i]} - area: {contour_area_sqdeg:.2f}sqdeg')

            ax.scatter(min_ra, min_dec, s=20, marker='*', color='black', label=r'scan best-fit', zorder=2)
            ax.legend(leg_element, leg_labels, loc='lower right', fontsize=8, scatterpoints=1, ncol=2)

            LOGGER.info(f"Contour Area (90%): {contour_area_sqdeg} degrees (cartesian) {contour_area_sqdeg * np.cos(min_dec)**2} degrees (scaled)")
            
            x_width = 1.6 * np.sqrt(contour_area_sqdeg)
            LOGGER.info(f"x width is {x_width}")
            
            if np.isnan(x_width):
                # this get called only when contour_area / x_width is NaN so possibly never invoked in typical situations
                
                raise RuntimeError("Estimated area / width is NaN and the fallback logic for this scenario is no longer supported. If you encounter this error raise an issue to SkyReader.")
                
                # mypy error: "QuadContourSet" has no attribute "allsegs"  [attr-defined]
                # this attribute is likely deprecated but this scenario is rarely (if ever) hit
                # original code is kept commented for the time being

                # note: contour_set is re-assigned at every iteration of the loop on
                # contour_levels, contour_colors, so this effectively corresponds to the
                # the last contour_set
                
                # x_width = 1.6*(max(contour_set.allsegs[i][0][:,0]) - min(contour_set.allsegs[i][0][:,0]))


            y_width = 0.5 * x_width

            lower_x = max(min_ra  - x_width*np.pi/180., 0.)
            upper_x = min(min_ra  + x_width*np.pi/180., 2 * np.pi)
            lower_y = max(min_dec -y_width*np.pi/180., -np.pi/2.)
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

            ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=tick_label_grid))
            ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=tick_label_grid))

        # cb.ax.xaxis.labelpad = -8
        # workaround for issue with viewers, see colorbar docstring
        # mypy compliance: since cb.solids could be None, we check that it is actually
        #   a valid object before accessing it
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
        effects: List[patheffects.AbstractPathEffect] = [patheffects.withStroke(linewidth=1.1, foreground='w')]
        for artist in ax.findobj(text.Text):
            # mypy error: Argument 1 to "set_path_effects" of "Artist" has incompatible type "list[withStroke]"; expected "list[AbstractPathEffect]"  [arg-type]
            artist.set_path_effects(effects)

        # remove white space around figure
        spacing = 0.01
        if not dozoom:
            fig.subplots_adjust(bottom=spacing, top=1.-spacing, left=spacing+0.04, right=1.-spacing)
        else:
            fig.subplots_adjust(bottom=spacing, top=0.92-spacing, left=spacing+0.1, right=1.-spacing)

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
            delta= sigma/180.0*np.pi
            step = 1./np.sin(delta)/10.
            bins = int(360./step)
            Theta = np.zeros(bins+1, dtype=np.double)
            Phi = np.zeros(bins+1, dtype=np.double)
            # define the contour
            for j in range(0,bins) :
                phi = j*step/180.*np.pi
                vx = np.cos(phi)*np.sin(ra)*np.sin(delta) + np.cos(ra)*(np.cos(delta)*np.sin(dec) + np.cos(dec)*np.sin(delta)*np.sin(phi))
                vy = np.cos(delta)*np.sin(dec)*np.sin(ra) + np.sin(delta)*(-np.cos(ra)*np.cos(phi) + np.cos(dec)*np.sin(ra)*np.sin(phi))
                vz = np.cos(dec)*np.cos(delta) - np.sin(dec)*np.sin(delta)*np.sin(phi)
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
                           circular_err90=0.7):
        """Uses healpy to plot a map."""

        def bounding_box(ra, dec, theta, phi):
            shift = ra-180

            ra_plus = np.max((np.degrees(phi)-shift)%360) - 180
            ra_minus = np.min((np.degrees(phi)-shift)%360) - 180
            dec_plus = (np.pi/2-np.min(theta))*180./np.pi - dec
            dec_minus = (np.pi/2-np.max(theta))*180./np.pi - dec
            return ra_plus, ra_minus, dec_plus, dec_minus

        dpi = self.PLOT_DPI_ZOOMED

        lonra=[-10.,10.]
        latra=[-10.,10.]

        event_metadata = result.get_event_metadata()
        unique_id = f'{str(event_metadata)}_{result.get_nside_string()}'
        plot_title = f"Run: {event_metadata.run_id} Event {event_metadata.event_id}: Type: {event_metadata.event_type} MJD: {event_metadata.mjd}"

        nsides = result.nsides
        LOGGER.info(f"available nsides: {nsides}")

        if systematics is not True:
            plot_filename = unique_id + ".plot_zoomed_wilks.pdf"
        else:
            plot_filename = unique_id + ".plot_zoomed.pdf"
        LOGGER.info(f"saving plot to {plot_filename}")

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
            grid_dec_list.append(dec); grid_ra_list.append(ra)
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

        LOGGER.info(f"min  RA: {min_ra *180./np.pi} deg, {min_ra*12./np.pi} hours.")
        LOGGER.info(f"min Dec: {min_dec * 180./np.pi} deg")

        # renormalize
        grid_value = grid_value - min_value
        min_value = 0.

        # show 2 * delta_LLH
        grid_value = grid_value * 2.

        # Do same for the healpy map
        equatorial_map[np.isinf(equatorial_map)] = np.nan
        equatorial_map -= np.nanmin(equatorial_map)
        equatorial_map *= 2.

        LOGGER.info(f"preparing plot: {plot_filename}...")

        cmap = self.PLOT_COLORMAP
        cmap.set_under('w')
        cmap.set_bad(alpha=1., color=(1.,0.,0.)) # make NaNs bright red

        # Calculate the contours
        if systematics:
            # from Pan-Starrs event 127852
            contour_levels = (np.array([22.2, 64.2])+min_value) # these are values determined from MC by Will on the TS (2*LLH)
            contour_labels = [r'50% (IC160427A syst.)', r'90% (IC160427A syst.)']
            contour_colors=['k', 'r']
        else:
            # Wilks
            contour_levels = (np.array([1.39, 4.61, 11.83, 28.74])+min_value)[:3]
            contour_labels = [r'50%', r'90%', r'3$\sigma$', r'5$\sigma$'][:3]
            contour_colors=['k', 'r', 'g', 'b'][:3]

        sample_points = np.array([np.pi/2 - grid_dec, grid_ra]).T
        
        # Call meander module to find contours
        if not circular:
            contours_by_level = meander.spherical_contours(sample_points,
                grid_value, contour_levels
                )
        if circular:
            sigma50 = np.deg2rad(circular_err50)
            sigma90 = np.deg2rad(circular_err90)
            Theta50, Phi50 = self.circular_contour(min_ra, min_dec, sigma50, nside)
            Theta90, Phi90 = self.circular_contour(min_ra, min_dec, sigma90, nside)
            contour50 = np.dstack((Theta50,Phi50))
            contour90 = np.dstack((Theta90,Phi90))
            contours_by_level = [contour50, contour90]
            
        # Check for RA values that are out of bounds
        for level in contours_by_level:
            for contour in level:
                contour.T[1] = np.where(contour.T[1] < 0.,
                    contour.T[1] + 2.*np.pi, contour.T[1]
                    )


        # Find the rough extent of the contours to bound the plot
        contours = contours_by_level[-1]
        ra = min_ra * 180./np.pi
        dec = min_dec * 180./np.pi
        theta, phi = np.concatenate(contours_by_level[-1]).T
        ra_plus, ra_minus, dec_plus, dec_minus = bounding_box(ra, dec, theta, phi)
        ra_bound = min(15, max(3, max(-ra_minus, ra_plus)))
        dec_bound = min(15, max(2, max(-dec_minus, dec_plus)))
        lonra = [-ra_bound, ra_bound]
        latra = [-dec_bound, dec_bound]

        #Begin the figure
        plt.clf()
        # Rotate into healpy coordinates
        lon, lat = np.degrees(min_ra), np.degrees(min_dec)
        healpy.cartview(map=equatorial_map, title=plot_title,
            min=0., #min 2DeltaLLH value for colorscale
            max=40., #max 2DeltaLLH value for colorscale
            rot=(lon,lat,0.), cmap=cmap, hold=True,
            cbar=None, lonra=lonra, latra=latra,
            unit=r"$-2 \Delta \ln (L)$",
            )

        fig = plt.gcf()
        ax = plt.gca()
        image = ax.get_images()[0]
        # Place colorbar by hand
        cb = fig.colorbar(image, ax=ax, orientation='horizontal', aspect=50)
        cb.ax.xaxis.set_label_text(r"$-2 \Delta \ln (L)$")

        # Plot the best-fit location
        # This requires some more coordinate transformations
        healpy.projplot(np.pi/2 - min_dec, min_ra,
            '*', ms=5, label=r'scan best fit', color='black', zorder=2)

        # Plot the contours
        contour_areas=[]
        for contour_label, contour_color, contours in zip(
            contour_labels, contour_colors, contours_by_level):
            contour_area = 0.
            for contour in contours:
                _ = contour.copy()
                _[:,1] += np.pi-np.radians(ra)
                _[:,1] %= 2*np.pi
                contour_area += self.calculate_area(_)
            contour_area_sqdeg = abs(contour_area) * (180.*180.)/(np.pi*np.pi) # convert to square-degrees
            contour_areas.append(contour_area_sqdeg)
            contour_label = contour_label + ' - area: {0:.2f} sqdeg'.format(
                contour_area_sqdeg)
            first = True
            for contour in contours:
                theta, phi = contour.T
                if first:
                    healpy.projplot(theta, phi, linewidth=2, c=contour_color,
                        label=contour_label)
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
        hp_ticklabels(zoom=True, lonra=lonra, latra=latra,
            rot=(lon,lat,0),
            bounds=(lower_lon, upper_lon, lower_lat, upper_lat))

        if plot_4fgl:
            # Overlay 4FGL sources
            plot_catalog(equatorial_map, cmap, lower_ra, upper_ra, lower_dec, upper_dec)

        # Approximate contours as rectangles
        ra = min_ra * 180./np.pi
        dec = min_dec * 180./np.pi
        for l, contours in enumerate(contours_by_level[:2]):
            ra_plus = None
            theta, phi = np.concatenate(contours).T
            ra_plus, ra_minus, dec_plus, dec_minus = bounding_box(ra, dec, theta, phi)
            contain_txt = "Approximating the {0}% error region as a rectangle, we get:".format(["50", "90"][l]) + " \n" + \
                          "\t RA = {0:.2f} + {1:.2f} - {2:.2f}".format(
                              ra, ra_plus, np.abs(ra_minus)) + " \n" + \
                          "\t Dec = {0:.2f} + {1:.2f} - {2:.2f}".format(
                              dec, dec_plus, np.abs(dec_minus))
            # This is actually an output and not a logging info.
            # TODO: we should wrap this in an object, return and log at the higher level.
            print(contain_txt)

        if plot_bounding_box:
            bounding_ras_list, bounding_decs_list = [], []
            # lower bound
            bounding_ras_list.extend(list(np.linspace(ra+ra_minus,
                ra+ra_plus, 10)))
            bounding_decs_list.extend([dec+dec_minus]*10)
            # right bound
            bounding_ras_list.extend([ra+ra_plus]*10)
            bounding_decs_list.extend(list(np.linspace(dec+dec_minus,
                dec+dec_plus, 10)))
            # upper bound
            bounding_ras_list.extend(list(np.linspace(ra+ra_plus,
                ra+ra_minus, 10)))
            bounding_decs_list.extend([dec+dec_plus]*10)
            # left bound
            bounding_ras_list.extend([ra+ra_minus]*10)
            bounding_decs_list.extend(list(np.linspace(dec+dec_plus,
                dec+dec_minus,10)))
            # join end to beginning
            bounding_ras_list.append(bounding_ras_list[0])
            bounding_decs_list.append(bounding_decs_list[0])

            bounding_ras: np.ndarray = np.asarray(bounding_ras_list)
            bounding_decs: np.ndarray = np.asarray(bounding_decs_list)
            bounding_phi = np.radians(bounding_ras)
            bounding_theta = np.pi/2 - np.radians(bounding_decs)
            bounding_contour = np.array([bounding_theta, bounding_phi])
            bounding_contour_area = 0.
            bounding_contour_area = abs(self.calculate_area(bounding_contour.T))
            bounding_contour_area *= (180.*180.)/(np.pi*np.pi) # convert to square-degrees
            contour_label = r'90% Bounding rectangle' + ' - area: {0:.2f} sqdeg'.format(
                bounding_contour_area)
            healpy.projplot(bounding_theta, bounding_phi, linewidth=0.75,
                c='r', linestyle='dashed', label=contour_label)

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
                LOGGER.info("Dumping to {self.output_dir / savename}")
                ascii.write(tab, self.output_dir / savename, overwrite=True)
            except OSError as err:
                LOGGER.error("OS Error prevented contours from being written, maybe a memory issue. Error is:\n{err}")

        uncertainty = [(ra_minus, ra_plus), (dec_minus, dec_plus)]
        fits_header = format_fits_header(
            (event_metadata.run_id, event_metadata.event_id, event_metadata.event_type),
            0,
            np.degrees(min_ra),
            np.degrees(min_dec),
            uncertainty,
        )
        mmap_nside = healpy.get_nside(equatorial_map)

        # Plot the original online reconstruction location
        if np.sum(np.isnan([extra_ra, extra_dec, extra_radius])) == 0:

            # dist = angular_distance(minRA, minDec, extra_ra * np.pi/180., extra_dec * np.pi/180.)
            # print("Millipede best fit is", dist /(np.pi * extra_radius/(1.177 * 180.)), "sigma from reported best fit")
        

            extra_ra_rad = np.radians(extra_ra)
            extra_dec_rad = np.radians(extra_dec)
            extra_radius_rad = np.radians(extra_radius)
            extra_lon = extra_ra_rad
            extra_lat = extra_dec_rad

            healpy.projscatter(np.degrees(extra_lon), np.degrees(extra_lat),
                lonlat=True, c='m', marker='x', s=20, label=r'Reported online (50%, 90%)')
            for cont_lev, cont_scale, cont_col, cont_sty in zip(['50', '90.'], 
                    [1., 2.1459/1.177], ['m', 'm'], ['-', '--']):
                spline_contour = self.circular_contour(extra_ra_rad, extra_dec_rad,
                    extra_radius_rad*cont_scale, healpy.get_nside(equatorial_map))
                spline_lon = spline_contour[1]
                spline_lat = np.pi/2. - spline_contour[0]
                healpy.projplot(np.degrees(spline_lon), np.degrees(spline_lat), 
                    lonlat=True, linewidth=2., color=cont_col, 
                    linestyle=cont_sty)

        plt.legend(fontsize=6, loc="lower left")


        print("Contour Area (50%):", contour_areas[0], "degrees (scaled)")
        print("Contour Area (90%):", contour_areas[1], "degrees (scaled)")


        # Dump the whole contour
        pickle_path = self.output_dir / (unique_id + ".contour.pkl")
        LOGGER.info(f"Saving contour to {pickle_path}")
        with open(pickle_path, "wb") as f:
            pickle.dump(saving_contours, f)

        healpy.write_map(self.output_dir / f"{unique_id}.skymap_nside_{mmap_nside}.fits.gz",
            equatorial_map, coord = 'C', column_names = ['2LLH'],
            extra_header = fits_header, overwrite=True)

        # Save the figure
        LOGGER.info(f"Saving: {plot_filename}...")
        #ax.invert_xaxis()
        fig.savefig(self.output_dir / plot_filename, dpi=dpi, transparent=True, bbox_inches='tight')

        LOGGER.info("done.")

        plt.close()
