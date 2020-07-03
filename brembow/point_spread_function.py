import scipy


class PointSpreadFunction:
    def __init__(self, *args, **kwargs):
        pass

    def apply_psf(self, *args, **kwargs):
        pass


class GaussianPSF(PointSpreadFunction):
    def __init__(self, intensity, sigma):
        self.intensity = intensity
        self.sigma = sigma

    def apply_psf(self, point_image):
        scipy.ndimage.gaussian_filter(
            point_image,
            self.sigma,
            output=point_image,
            truncate=3.0)
        point_image *= self.intensity
