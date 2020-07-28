import numpy as np
from brembow import (simulate_random_cages,
                     render_points,
                     render_cage,
                     render_cage_distribution,
                     simulate_cages)
from brembow import Cage
from brembow import Volume
from brembow import PointSpreadFunction, GaussianPSF
import zarr
import gunpowder as gp
import random


class SimulateCages(gp.BatchFilter):
    ''' Gunpowder node to simulate cages on the fly within a raw array. Requires a
    segmentation corresponding to the raw array. Renders a random type of
    cage with a random density within each segment.

        Args:

            raw (ArrayKey): ArrayKey that points to the volume to render to.

            seg (Volume object): A segmentation of the volume. The segmentation
            is expected to be int valued with values between 1 and n. 0 will be
            treated as background.

            res (float): The resolution of volume.

            psf (PointSpreadFunction): The PSF to use to render points.

            density_rang (tuple of floats): The min and max density to
            uniformly choose from.

            cages (list of Cages): A list of cages to randomly select from.

        '''
    def __init__(
            self,
            raw,
            seg,
            res,
            psf,
            density_range,
            cages):

        self.raw = raw
        self.seg = seg
        self.res = res
        self.cage_ids = {}
        self.densities = {}
        self.psf = psf
        self.min_density, self.max_density = density_range
        self.cages = cages

        id_list = np.unique(self.seg.data)
        id_list = id_list[np.nonzero(id_list)]

        for id_element in id_list:
            self.cage_ids[id_element] = random.choice(self.cages)
            self.densities[id_element] = random.uniform(self.min_density,
                                                        self.max_density)

        # no need for setup since we are modifying the image in-place

    def prepare(self, request):
        # provide segmentation
        # currently, all of tiny_raw is ROI
        # how to change this to be specific to segment ID?
        roi = request[self.raw].roi

        print("ROI")
        print(roi)

        pass

    def process(self, batch, request):
        data = batch[self.raw].data

        print("DATA")
        print(data)

        volume = simulate_cages(Volume(data, self.res),
                                self.seg, self.cage_ids,
                                self.densities,
                                self.psf)
        batch[self.raw].data = volume.data


datafile = zarr.open(
    '/Users/ekarlekar/Documents/Funke/data/cropped_sample_A.zarr', 'r')
tiny_seg = datafile['tiny_segmentation'][:]
resolution = datafile['tiny_raw'].attrs['resolution']
seg = Volume(tiny_seg, resolution)

cage1 = Cage("/Users/ekarlekar/Documents/Funke/data/example_cage")

psf = GaussianPSF(intensity=0.125, sigma=(1.0, 1.0))
min_density = 2e-5
max_density = 2e-5
tiny_raw = gp.ArrayKey('tiny_raw')
source = gp.ZarrSource(
    '/Users/ekarlekar/Documents/Funke/data/cropped_sample_A.zarr',
    {tiny_raw: 'tiny_raw'},
    {tiny_raw: gp.ArraySpec(interpolatable=True, voxel_size=resolution)}
)
print("SOURCE")
print(source)


normalize = gp.Normalize(tiny_raw)
pipeline = (source + normalize + SimulateCages(tiny_raw,
                                               seg,
                                               resolution,
                                               psf,
                                               (min_density, max_density),
                                               [cage1]))

print("PIPELINE")
print(pipeline)

request = gp.BatchRequest()

# how to change this to be specific to segment ID?
request[tiny_raw] = gp.Roi((0, 0, 0), (400, 4000, 4000))

print("REQUEST")
print(request)

with gp.build(pipeline):
    batch = pipeline.request_batch(request)

with zarr.open('testV.zarr', 'w') as f:
    f['render'] = batch[tiny_raw].data
    f['render'].attrs['resolution'] = resolution

print("done")
