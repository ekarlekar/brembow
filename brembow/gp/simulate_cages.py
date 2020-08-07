import numpy as np
from brembow import simulate_random_cages
from brembow import Volume
import gunpowder as gp


class SimulateCages(gp.BatchFilter):
    ''' Gunpowder node to simulate cages on the fly within a raw array. Requires a
    segmentation corresponding to the raw array. Renders a random type of
    cage with a random density within each segment.

        Args:

            raw (ArrayKey): ArrayKey that points to the volume to render to.

            seg (ArrayKey): ArrayKey that points to a segmentation of the
            volume. The segmentation is expected to be int valued with values
            between 1 and n. 0 will be treated as background.

            psf (PointSpreadFunction): The PSF to use to render points.

            density_range (tuple of floats): The min and max density to
            uniformly choose from.

            cages (list of Cages): A list of cages to randomly select from.
        '''
    def __init__(
            self,
            raw,
            seg,
            cage_map,
            density_map,
            psf,
            density_range,
            cages,
            no_cage_probability=0.0):

        # High-level summary:
        #
        # 1. request raw and segmentation (of same size)
        # 2. change raw (in-place) by rendering cages
        # 3. provide two maps: cages and densities
        #
        # Sizes of arrays and requests:
        #
        # In general, a request to this node could look like this:
        #
        #  raw      :  |---------------------|
        #  cages map:        |---------|
        #
        # (maybe also:)
        #  seg      :        |---------|
        #  dens. map:        |---------|
        #
        # To produce the maps and update the raw array, our node needs:
        #
        #  raw      :  |---------------------|
        #  seg      :  |---------------------|

        self.raw = raw
        self.seg = seg
        self.cage_map = cage_map
        self.density_map = density_map
        self.psf = psf
        self.min_density, self.max_density = density_range
        self.cages = cages
        self.no_cage_probability = no_cage_probability

    def setup(self):

        # we provide cage maps everywhere where we have a segmentation:
        roi = self.spec[self.seg].roi.copy()
        voxel_size = self.spec[self.seg].voxel_size
        self.provides(
            self.cage_map,
            gp.ArraySpec(roi=roi, dtype=np.uint16, voxel_size=voxel_size))

        # same for the density map
        roi = self.spec[self.seg].roi.copy()
        self.provides(
            self.density_map,
            gp.ArraySpec(roi=roi, dtype=np.float32, voxel_size=voxel_size))

    def prepare(self, request):

        roi = request[self.raw].roi

        deps = gp.BatchRequest()

        deps[self.raw] = roi
        deps[self.seg] = roi

        return deps

    def process(self, batch, request):

        # get the raw and segmentation arrays from the current batch
        raw = batch[self.raw]
        seg = batch[self.seg]

        print(f"RAW: {raw}")
        print(f"SEG: {seg}")

        # simulate cages, return brembow volumes for raw, cages, and density
        simulated_raw = Volume(raw.data, raw.spec.voxel_size)
        cage_map, density_map = simulate_random_cages(
            simulated_raw,
            Volume(seg.data, seg.spec.voxel_size),
            self.cages,
            self.min_density,
            self.max_density,
            self.psf,
            True,
            True,
            self.no_cage_probability)

        # create array specs for new gunpowder arrays
        raw_spec = batch[self.raw].spec.copy()
        cage_map_spec = batch[self.seg].spec.copy()
        cage_map_spec.dtype = np.uint64
        density_map_spec = batch[self.seg].spec.copy()
        density_map_spec.dtype = np.float32

        # create arrays and crop to requested size
        print(cage_map_spec)
        cage_map_array = gp.Array(data=cage_map, spec=cage_map_spec)
        cage_map_array = cage_map_array.crop(request[self.cage_map].roi)
        density_map_array = gp.Array(data=density_map, spec=density_map_spec)
        density_map_array = density_map_array.crop(
                                            request[self.density_map].roi)

        # create a new batch with processed arrays
        processed = gp.Batch()
        processed[self.raw] = gp.Array(data=simulated_raw.data, spec=raw_spec)
        processed[self.cage_map] = cage_map_array
        processed[self.density_map] = density_map_array

        return processed
