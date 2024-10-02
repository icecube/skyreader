import numpy as np
from typing import List


# Calculates are using Gauss-Green theorem / shoelace formula
# TODO: vectorize using numpy.
# Note: in some cases the argument is not a np.ndarray so one has
# to convert the data series beforehand.
def calculate_area(vs) -> float:
    a = 0
    x0, y0 = vs[0]
    for [x1, y1] in vs[1:]:
        dx = np.cos(x1)-np.cos(x0)
        dy = y1-y0
        a += 0.5*(y0*dx - np.cos(x0)*dy)
        x0 = x1
        y0 = y1
    return a


# Given a list of contours by level, returns the areas of the contours
def get_contour_areas(contours_by_level_list, min_ra) -> List[float]:
    contour_areas = []
    ra = min_ra * 180./np.pi
    for contours in contours_by_level_list:
        contour_area = 0.
        for contour in contours:
            _ = contour.copy()
            _[:,1] += np.pi-np.radians(ra)
            _[:,1] %= 2*np.pi
            contour_area += abs(calculate_area(_))
        # convert to square-degrees
        contour_area_sqdeg = abs(contour_area) * (180.*180.)/(np.pi*np.pi)
        contour_areas.append(contour_area_sqdeg)
    return contour_areas
