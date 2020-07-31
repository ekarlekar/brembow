from brembow import Cage
from brembow import PointSpreadFunction, GaussianPSF
import zarr
import gunpowder as gp
from brembow.gp import SimulateCages

datafile = zarr.open(
    '/Users/ekarlekar/Documents/Funke/data/cropped_sample_A.zarr', 'r')
tiny_seg = datafile['tiny_segmentation'][:]
resolution = datafile['tiny_raw'].attrs['resolution']


cage1 = Cage("/Users/ekarlekar/Documents/Funke/data/example_cage")

psf = GaussianPSF(intensity=0.125, sigma=(1.0, 1.0))
min_density = 2e-5
max_density = 2e-5
tiny_raw = gp.ArrayKey('TINY_RAW')
seg = gp.ArrayKey('TINY_SEGMENTATION')
print("RESOLUTION")
print(resolution)
source = gp.ZarrSource(
    '/Users/ekarlekar/Documents/Funke/data/cropped_sample_A.zarr',
    {tiny_raw: 'tiny_raw', seg: 'tiny_segmentation'},
    {
        tiny_raw: gp.ArraySpec(interpolatable=True, voxel_size=resolution),
        seg: gp.ArraySpec(interpolatable=False, voxel_size=resolution)
    }
)

print("SOURCE")
print(source)

out_cage_map = gp.ArrayKey('OUT_CAGE_MAP')
out_density_map = gp.ArrayKey('OUT_DENSITY_MAP')
normalize = gp.Normalize(tiny_raw)
pipeline = (source + normalize + SimulateCages(tiny_raw,
                                               seg,
                                               out_cage_map,
                                               out_density_map,
                                               psf,
                                               (min_density, max_density),
                                               [cage1]))

print("PIPELINE")
print(pipeline)

request = gp.BatchRequest()

# how to change this to be specific to segment ID?
request[tiny_raw] = gp.Roi((0, 0, 0), (400, 4000, 4000))
request[out_cage_map] = gp.Roi((0, 0, 0), (400, 4000, 4000))
request[out_density_map] = gp.Roi((0, 0, 0), (400, 4000, 4000))

print("REQUEST")
print(request)

with gp.build(pipeline):
    batch = pipeline.request_batch(request)

with zarr.open('testV.zarr', 'w') as f:
    f['render'] = batch[tiny_raw].data
    f['render'].attrs['resolution'] = resolution

print("done")
