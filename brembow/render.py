from .volume import Volume
from funlib.segment.arrays import replace_values
import numpy as np
import random


def render_points(
        resolution,
        image,
        locations,
        intensities,
        point_spread_function):
    '''Render a list of points with a Gaussian point-spread-function into a 2D
    image. This rendering is subtractive with respect to the contents already
    present in the image.

    Args:

        resolution (tuple of two floats): The yx resolution of the image.

        image (ndarray): The image to render to. The image is expected to real
        valued with values between 0 and 1.

        locations (list of tuple of floats): The locations (in pixels)
        where to place the points.

        intensities (array of float): The intensities with which to render the
        points given by ``locations``.

        point_spread_function (PointSpreadFunction): The point-spread function
        to use.
    '''

    assert locations.shape[0] == intensities.shape[0]

    # for each point, change point image to 1.0
    locations = np.array(locations)/resolution
    locations = locations.astype(int)

    # limit image to area affected by locations
    radius = np.ceil(point_spread_function.get_radius()/resolution).astype(int)
    bounding_box_min = np.min(locations, axis=0) - radius
    bounding_box_max = np.max(locations, axis=0) + radius
    bounding_box_min, bounding_box_max = np.clip(
        [bounding_box_min, bounding_box_max],
        [0, 0],
        [image.shape[0] - 1, image.shape[1] - 1])
    image = image[
        bounding_box_min[0]:bounding_box_max[0] + 1,
        bounding_box_min[1]:bounding_box_max[1] + 1]

    locations -= bounding_box_min
    x_values = locations[:, 0]
    y_values = locations[:, 1]
    coords = [tuple(x_values), tuple(y_values)]

    point_image = np.zeros_like(image)

    # x, y = int(location[0]/resolution), int(location[1]/resolution)
    np.add.at(point_image, tuple(coords), intensities)

    # blurs image according to appropriate PSF
    point_spread_function.apply_psf(point_image)

    # subract point image from original image
    image -= point_image

    # ensure values are still between 0 and 1
    np.clip(image, 0.0, 1.0, out=image)


def render_cage(volume, location, cage, fm_intensity, point_spread_function):
    '''Render a cage with a Gaussian point-spread-function into a 3D volume.
    This rendering is subtractive with respect to the contents already present
    in the volume.

    Args:

        volume (Volume object): The volume to render to. The volume is
        expected to be real valued with values between 0 and 1.

        location (tuple of floats): The location (in pixels) where to place the
        cage.

        cage (`class:Cage`): The cage to render.

        fm_intensity (float): Render intensity for element 100 (Fermium), to be
        used as reference point for cubic intensity transfer function.

        point_spread_function (PointSpreadFunction): The PSF to use to render
        points.
    '''

    # normalizing points in relation to cage loc
    pre_norm_cage_loc = np.array(cage.get_locations())
    atom_locations = pre_norm_cage_loc + location
    atomic_numbers = cage.get_atomic_numbers()

    depth, height, width = volume.data.shape
    resolution = volume.resolution

    for zplane in range(0, depth):
        zbegin, zend = zplane*resolution[0], (zplane + 1)*resolution[0]
        ybegin, yend = 0, height*resolution[1]
        xbegin, xend = 0, height*resolution[2]
        visible_atoms = (
            (atom_locations[:, 0] >= zbegin) &
            (atom_locations[:, 0] < zend) &
            (atom_locations[:, 1] >= ybegin) &
            (atom_locations[:, 1] < yend) &
            (atom_locations[:, 2] >= xbegin) &
            (atom_locations[:, 2] < xend)
        )
        valid_locs = atom_locations[visible_atoms]
        valid_atomic_numbers = atomic_numbers[visible_atoms]

        if len(valid_locs) == 0:
            continue

        valid_intensities = compute_intensities(
            fm_intensity,
            valid_atomic_numbers)

        render_points(volume.resolution[1:3],
                      volume.data[zplane, :, :],
                      valid_locs[:, 1:3],
                      valid_intensities,
                      point_spread_function)


def compute_intensities(fm_intensity, atomic_numbers):

    # 0 -> 0
    # 100 -> FmI  (intensity of Fm)
    #
    # y = α x^3
    # FmI = α 100^3 ⇒ α = FmI/100^3

    return fm_intensity/(100**3) * atomic_numbers**3


def render_cage_distribution(
        volume,
        cage,
        fm_intensity,
        point_spread_function,
        density,
        mask=None):
    '''Renders randomly oriented copies of the given cage with the given
    density into volume.

    Args:

        volume (Volume): The volume to render to. The volume is expected to be
        real valued with values between 0 and 1.

        cage (`class:Cage`): The cage to render.

        sigma (float): The standard deviation of the Gaussian
        point-spread-function.

        fm_intensity (float): Render intensity for element 100 (Fermium), to be
        used as reference point for cubic intensity transfer function.

        point_spread_function (PointSpreadFunction): The PSF to use to render
        points.

        density (float): The density of cages in number of cages per cubic
        micron.

        mask (Volume, optional): If given, limit rendering to areas where mask
        is > 0.
    '''
    depth, height, width = volume.data.shape
    num_voxels = depth*height*width
    vol = num_voxels*np.prod(volume.resolution)
    num_expected_points = int(density*vol)

    size = volume.resolution * [depth, height, width]
    locations = np.random.random((num_expected_points, 3)) * size

    # filter locations by mask (if given)
    if(mask is not None):
        locations = filter_locations(volume, mask, locations)

    # for each location:
    for loc in locations:

        # randomly rotate cage (account for gimbal lock)
        cage.set_random_rotation()

        # render cage
        render_cage(volume, loc, cage, fm_intensity, point_spread_function)


def filter_locations(volume, mask, locations):
    '''Filter a list of locations (in world units) depending on whether they
    are part of the masked-in area.'''

    depth, height, width = volume.data.shape
    voxel_locations = locations/mask.resolution
    voxel_locations = voxel_locations.astype(np.int32)
    mask_depth, mask_height, mask_width = mask.data.shape
    voxel_locations[:, 0] = np.clip(voxel_locations[:, 0],
                                    0, mask_depth - 1)
    voxel_locations[:, 1] = np.clip(voxel_locations[:, 1],
                                    0, mask_height - 1)
    voxel_locations[:, 2] = np.clip(voxel_locations[:, 2],
                                    0, mask_width - 1)
    indices = (
            voxel_locations[:, 0]*height*width +
            voxel_locations[:, 1]*width +
            voxel_locations[:, 2])
    valid = mask.data.flatten()[indices].astype(np.bool)

    return locations[valid]


def simulate_cages(
        volume,
        segmentation,
        cages,
        densities,
        fm_intensity,
        point_spread_function):
    '''Render different cages with different densities into each segment.

    Args:

        volume (Volume): The volume to render to. The volume is expected to be
        real valued with values between 0 and 1.

        segmentation (Volume): A segmentation of the volume. The segmentation
        is expected to be int valued with values between 1 and n. 0 will be
        treated as background.

        cages (dict from int -> `class:Cage`): The cages to render per segment.

        densities (dict from int -> float): The density of cages per segment.

        fm_intensity (float): Render intensity for element 100 (Fermium), to be
        used as reference point for cubic intensity transfer function.

        point_spread_function (PointSpreadFunction): The PSF to use to render
        points.
    '''
    # which IDs do we have in the segmentation? (can we use numpy?)
    assert(volume.data.min() >= 0 and volume.data.max() <= 1)
    id_list = np.unique(segmentation.data)
    id_list = id_list[np.nonzero(id_list)]

    # for each segment ID:
    for id_element in id_list:
        # find appropriate cage and density
        cage = cages.get(id_element)
        density = densities.get(id_element)

        if cage is None:

            continue

        if density is None:
            print(f"WARNING: segment ID {id_element} does not have a density "
                  "associated")
            continue

        # create a binary mask
        mask_data = np.where(segmentation.data == id_element, 1, 0)
        mask = Volume(mask_data, segmentation.resolution)

        # call render_cage_distribution with the correct cage and density
        render_cage_distribution(
            volume,
            cage,
            fm_intensity,
            point_spread_function,
            density,
            mask)


def simulate_random_cages(
        volume,
        segmentation,
        cages,
        min_density,
        max_density,
        fm_intensity,
        point_spread_function,
        return_cage_map=False,
        return_density_map=False,
        no_cage_probability=0.0):
    '''Randomly render cages with a range of densities for each segment into a
    volume.

    Args:

        volume (Volume): The volume to render to. The volume is expected to be
        real valued with values between 0 and 1.

        segmentation (Volume): A segmentation of the volume. The segmentation
        is expected to be int valued with values between 1 and n. 0 will be
        treated as background.

        cages (list of Cages): A list of cages to randomly select from.

        min_density, max_density (float): The minimum and maximum density to
        uniformly choose from.

        fm_intensity (float): Render intensity for element 100 (Fermium), to be
        used as reference point for cubic intensity transfer function.

        point_spread_function (PointSpreadFunction): The PSF to use to render
        points.

        return_cage_map (bool): Return a map of which segment contains which
        type of cage (as an integer).

        return_density_map (bool): Return a map of the cage densities per
        segment.

        no_cage_probability (float): The probability of expressing no cage, per
        segment.
    '''
    assert(volume.data.min() >= 0 and volume.data.max() <= 1)

    id_list = np.unique(segmentation.data)
    id_list = id_list[np.nonzero(id_list)]

    random_cages = {}
    random_densities = {}

    for id_element in id_list:
        test = random.random()

        if test > no_cage_probability:
            random_cages[id_element] = random.choice(cages)
            random_densities[id_element] = random.uniform(min_density, max_density)
        else:
            random_cages[id_element] = None
            random_densities[id_element] = 0


    simulate_cages(
        volume,
        segmentation,
        random_cages,
        random_densities,
        fm_intensity,
        point_spread_function)

    ret = ()

    if return_cage_map:

        # replace segmentation IDs with cage IDs
        cage_map = replace_values(
            segmentation.data,
            id_list,
            [
                random_cages[i].cage_id if random_cages[i] else 0
                for i in id_list
            ])

        ret = ret + (cage_map,)

    if return_density_map:

        densities = np.array(
            [random_densities[i] for i in id_list],
            dtype=np.float64)

        # (almost) the same for the density map:
        density_map = replace_values(
            segmentation.data.astype(np.uint64),
            id_list.astype(np.uint64),
            densities.view(np.uint64)).view(np.float64)
        density_map = density_map.astype(np.float32)

        ret = ret + (density_map,)

    if len(ret) > 0:
        return ret
