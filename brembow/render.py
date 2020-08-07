from .volume import Volume
from funlib.segment.arrays import replace_values
import numpy as np
import random


def render_points(resolution, image, locations, point_spread_function):
    '''Render a list of points with a Gaussian point-spread-function into a 2D
    image. This rendering is subtractive with respect to the contents already
    present in the image.

    Args:

        resolution (tuple of two floats): The yx resolution of the image.

        image (ndarray): The image to render to. The image is expected to real
        valued with values between 0 and 1.

        locations (list of tuple of floats): The locations (in pixels)
        where to place the points.

        point_spread_function (PointSpreadFunction): The point-spread function
        to use.
    '''

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
    point_image[tuple(coords)] = 1.0

    # blurs image according to appropriate PSF
    point_spread_function.apply_psf(point_image)

    # subract point image from original image
    image -= point_image

    # ensure values are still between 0 and 1
    np.clip(image, 0.0, 1.0, out=image)


def render_cage(volume, location, cage, point_spread_function):
    '''Render a cage with a Gaussian point-spread-function into a 3D volume.
    This rendering is subtractive with respect to the contents already present
    in the volume.

    Args:

        volume (Volume object): The volume to render to. The volume is
        expected to be real valued with values between 0 and 1.

        location (tuple of floats): The location (in pixels) where to place the
        cage.

        cage (`class:Cage`): The cage to render.

        sigma (float): The standard deviation of the Gaussian
        point-spread-function.
    '''

    # normalizing points in relation to cage loc
    pre_norm_cage_loc = np.array(cage.get_locations())
    cage_point_locs = pre_norm_cage_loc + location

    depth, height, width = volume.data.shape
    resolution = volume.resolution

    for zplane in range(0, depth):
        zbegin, zend = zplane*resolution[0], (zplane + 1)*resolution[0]
        ybegin, yend = 0, height*resolution[1]
        xbegin, xend = 0, height*resolution[2]
        valid_locs = cage_point_locs[(cage_point_locs[:, 0] >= zbegin) &
                                     (cage_point_locs[:, 0] < zend) &
                                     (cage_point_locs[:, 1] >= ybegin) &
                                     (cage_point_locs[:, 1] < yend) &
                                     (cage_point_locs[:, 2] >= xbegin) &
                                     (cage_point_locs[:, 2] < xend)]
        if len(valid_locs) == 0:
            continue

        render_points(volume.resolution[1:3],
                      volume.data[zplane, :, :],
                      valid_locs[:, 1:3],
                      point_spread_function)


def render_cage_distribution(volume, cage,
                             point_spread_function, density, mask=None):
    '''Renders randomly oriented copies of the given cage with the given
    density into volume.

    Args:

        volume (Volume): The volume to render to. The volume is expected to be
        real valued with values between 0 and 1.

        cage (`class:Cage`): The cage to render.

        sigma (float): The standard deviation of the Gaussian
        point-spread-function.

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

    print("len of locations: " + str(len(locations)))

    # for each location:
    for loc in locations:

        # randomly rotate cage (account for gimbal lock)
        cage.set_random_rotation()

        # render cage
        render_cage(volume, loc, cage, point_spread_function)


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


def simulate_cages(volume, segmentation,
                   cages, densities, point_spread_function):
    '''Render different cages with different densities into each segment.

    Args:

        volume (Volume): The volume to render to. The volume is expected to be
        real valued with values between 0 and 1.

        segmentation (Volume): A segmentation of the volume. The segmentation
        is expected to be int valued with values between 1 and n. 0 will be
        treated as background.

        cages (dict from int -> `class:Cage`): The cages to render per segment.

        densities (dict from int -> float): The density of cages per segment.

        sigma (float): The standard deviation of the Gaussian
        point-spread-function.
    '''
    # which IDs do we have in the segmentation? (can we use numpy?)
    assert(volume.data.min() >= 0 and volume.data.max() <= 1)
    id_list = np.unique(segmentation.data)
    id_list = id_list[np.nonzero(id_list)]

    print("len of id_list" + str(len(id_list)))
    count = 0
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
        print(str(density), str(cage))
        # call render_cage_distribution with the correct cage and density
        render_cage_distribution(volume, cage,
                                 point_spread_function, density, mask)
        if(count % 50 == 0):
            print("another 50 done", count)

        count += 1


def simulate_random_cages(
        volume,
        segmentation,
        cages,
        min_density,
        max_density,
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

        point_spread_function (PointSpreadFunction): The PSF to use to render
        points.
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
