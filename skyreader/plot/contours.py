from dataclasses import dataclass
import healpy  # type: ignore[import]
import numpy as np
from typing import ClassVar


class CircularContour:
    @staticmethod
    def circular_contour(ra_rad: float, dec_rad: float, radius_rad: float, nside: int):
        """For plotting circular contours on skymaps ra, dec, sigma all
        expected in radians."""
        # This transformation is likely reversed later. It can probably be removed.
        # Likely legacy of old Skymap Scanner design.
        dec_rad = np.pi / 2.0 - dec_rad
        step = 1.0 / np.sin(radius_rad) / 10.0
        # bins * step = 360 deg
        bins = int(360.0 / step)
        Theta = np.zeros(bins + 1, dtype=np.double)
        Phi = np.zeros(bins + 1, dtype=np.double)
        # Define the contour
        for j in range(0, bins):
            # phi runs over 0, 2 pi
            phi = j * step / 180.0 * np.pi
            vx = np.cos(phi) * np.sin(ra_rad) * np.sin(radius_rad) + np.cos(ra_rad) * (
                np.cos(radius_rad) * np.sin(dec_rad)
                + np.cos(dec_rad) * np.sin(radius_rad) * np.sin(phi)
            )
            vy = np.cos(radius_rad) * np.sin(dec_rad) * np.sin(ra_rad) + np.sin(
                radius_rad
            ) * (
                -np.cos(ra_rad) * np.cos(phi)
                + np.cos(dec_rad) * np.sin(ra_rad) * np.sin(phi)
            )
            vz = np.cos(dec_rad) * np.cos(radius_rad) - np.sin(dec_rad) * np.sin(
                radius_rad
            ) * np.sin(phi)
            idx = healpy.vec2pix(nside, vx, vy, vz)
            Theta[j], Phi[j] = healpy.pix2ang(nside, idx)

        # Close the contour
        Theta[bins], Phi[bins] = Theta[0], Phi[0]

        return Theta, Phi


@dataclass
class GaussianContour(CircularContour):
    ra_deg: float
    dec_deg: float
    radius_50: float
    levels: list[float] = [50.0, 90.0]
    title: str = ""
    color: str = "m"
    marker: str = "x"
    marker_size: int = 20

    CONTOUR_STYLES: ClassVar[dict[float, str]] = {50.0: "-", 90.0: "--"}
    SCALE_FACTORS: ClassVar[dict[float, float]] = {50.0: 1.177, 90.0: 2.1459}

    @property
    def dec_rad(self):
        return np.deg2rad(self.dec_deg)

    @property
    def ra_rad(self):
        return np.deg2rad(self.ra_deg)

    @property
    def sigma_deg(self):
        return self.radius_50 / self.SCALE_FACTORS[50]

    @property
    def sigma_rad(self):
        return np.rad2deg(self.sigma_deg)

    def get_style(self, containment: float):
        return self.CONTOUR_STYLES[containment]

    def sigma2radius(self, containment: float):
        # Converts the sigma (standard deviation) to a given containment radius
        # given a required % of containment.
        # We use hardcoded values but this can be dynamically calculated with
        # a chi2 PDF.
        return self.SCALE_FACTORS[containment]

    def generate_contour(self, nside: int, containment: float):
        """For plotting circular contours on skymaps ra, dec, sigma all
        expected in radians."""
        radius_rad = self.sigma_rad * self.sigma2radius(containment=containment)
        theta, phi = self.circular_contour(
            ra_rad=self.ra_rad, dec_rad=self.dec_rad, radius_rad=radius_rad, nside=nside
        )
        return theta, phi
