import numpy as np
from .volume import Volume


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
    # now between -1 and 1 to account for modified images from render_cage
    assert(image.min() >= 0 and image.max() <= 1)

    point_image = np.zeros_like(image)

    # for each point, change point image to 1.0
    locations = np.array(locations)/resolution
    locations = locations.astype(int)

    x_values = locations[:, 0]
    y_values = locations[:, 1]
    coords = [tuple(x_values), tuple(y_values)]

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
    # TODO: should be in points per micron (volume.resolution)
    num_expected_points = int(density*(depth*height*width*volume.resolution))

    locations = (
        np.random.random((num_expected_points, 3)) *
        [depth, height, width]
    )

    # filter locations by mask (if given)
    if(mask is not None):
        voxel_locations = locations/mask.resolution
        voxel_locations = voxel_locations.astype(np.int32)
        mask_depth, mask_height, mask_width = mask.data.shape
        voxel_locations[:, 0] = np.clip(voxel_locations[:, 0],
                                        0, mask_depth - 1)
        voxel_locations[:, 1] = np.clip(voxel_locations[:, 1],
                                        0, mask_height - 1)
        voxel_locations[:, 2] = np.clip(voxel_locations[:, 2],
                                        0, mask_width - 1)

        z_values = voxel_locations[:, 0]
        y_values = voxel_locations[:, 1]
        x_values = voxel_locations[:, 2]
        valid = [tuple(z_values), tuple(y_values), tuple(x_values)]

        locations = locations[tuple(valid)]

    # for each location:
    for loc in locations:

        # randomly rotate cage (account for gimbal lock)
        cage.set_random_rotation()

        # render cage
        render_cage(volume, loc, cage, point_spread_function)


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
    id_list = np.nonzero(np.unique(segmentation.data))

    # for each segment ID:
    for id_element in id_list:

        # find appropriate cage and density
        cage = cages.get(id_element)
        density = densities.get(id_element)

        if cage is None:
            print(f"WARNING: segment ID {id_element} does not have a cage "
                  "associated")
            continue

        if density is None:
            print(f"WARNING: segment ID {id_element} does not have a density "
                  "associated")
            continue

        # create a binary mask
        mask_data = np.where(segmentation.data == id_element, 1, 0)
        mask = Volume(mask_data, segmentation.resolution)

        # call render_cage_distribution with the correct cage and density
        render_cage_distribution(volume, cage,
                                 point_spread_function, density, mask)
