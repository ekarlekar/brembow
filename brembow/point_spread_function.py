import numpy as np
from scipy.ndimage import gaussian_filter


class PointSpreadFunction:
    def __init__(self, *args, **kwargs):
        pass

    def apply_psf(self, *args, **kwargs):
        pass

    def get_radius(self):
        '''Return the radius (in world units) of this PSF'''
        pass

class GaussianPSF(PointSpreadFunction):
    def __init__(self, sigma):
        self.sigma = np.array(sigma)

    def apply_psf(self, point_image):
        gaussian_filter(
            point_image,
            self.sigma,
            output=point_image,
            truncate=3.0)

    def get_radius(self):
        return 3.0 * self.sigma
