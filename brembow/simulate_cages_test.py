import numpy as np
from brembow import simulate_random_cages, render_points, render_cage, render_cage_distribution, simulate_cages
from brembow import Cage
from brembow import Volume
from brembow import PointSpreadFunction, GaussianPSF
import zarr
import timeit
import os
print("CURRENT DIRECTORY: {}".format(os.getcwd()))
start_time = timeit.default_timer()
# cage_points = []
# for z in range(0,5):
#     for y in range(0,5):
#         for x in range (0,5):
#             cage_points.append([z,y,x])
datafile = zarr.open('/Users/ekarlekar/Documents/Funke/data/cropped_sample_A.zarr', 'r')
tiny_raw = datafile['tiny_raw'][:]/255.0
tiny_seg = datafile['tiny_segmentation'][:]
raw = Volume(tiny_raw, [40.0, 4.0, 4.0])
seg = Volume(tiny_seg, [40.0, 4.0, 4.0])
print("raw min:" + str(raw.data.min()))
print("raw max:" + str(raw.data.max()))
print("seg min:" + str(seg.data.min()))
print("seg max:" + str(seg.data.max()))
print(tiny_raw.shape)
print(tiny_seg.shape)
cage1 = Cage("/Users/ekarlekar/Documents/Funke/data/example_cage")
#cage1 = Cage(cage_points)
psf = GaussianPSF(0.5, 0.5)
min_density = 0.0001
max_density = 0.0005
#print(np.sort(cage1.locations, axis=0))
#render_cage_distribution(testvolume, cage1, psf, 0.0001)
simulate_random_cages(raw,
        seg,
        [cage1],
        min_density,
        max_density,
        psf)
#render_cage(testvolume, [200,200,200], cage1, psf)
elapsed = timeit.default_timer() - start_time
print("elapsed time: {}".format(elapsed))
with zarr.open('testvolume.zarr', 'w') as f:
    f['render'] = testvolume.data
print("done")
