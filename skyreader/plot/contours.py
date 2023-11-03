from dataclasses import dataclass
import healpy
import numpy as np
from typing import ClassVar

@dataclass
class GaussianContour:
    ra_deg: float
    dec_deg: float
    radius_50: float
    levels: list[float] = [50., 90.]
    title: str = ""
    color: str = 'm'
    marker: str = 'x'
    marker_size: int = 20


    CONTOUR_STYLES: ClassVar[dict[int, str]] = { 50 : '-', 90: '--' }
    SCALE_FACTORS: ClassVar[dict[int, float]] = { 50: 1.177, 90: 2.1459 }

    @property
    def dec_rad(self):
        return np.deg2rad(self.dec_deg)
    
    @property
    def ra_rad(self):
        return np.deg2rad(self.ra_deg)
    
    @property
    def sigma_deg(self):
        return self.radius_50 / self.SCALE_FACTORS[50]
    
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
            dec = np.pi/2. - self.dec_rad
            ra = self.ra_rad
            radius_deg = self.sigma_deg * self.sigma2radius(containment=containment)
            radius_rad = np.rad2deg(radius_deg)
            delta, step, bins = 0, 0, 0
            delta = radius_rad/180.0*np.pi
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