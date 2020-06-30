import numpy as np
import scipy
from .volume import Volume


def render_points(resolution, image, locations, sigma):
    '''Render a list of points with a Gaussian point-spread-function into a 2D
    image. This rendering is subtractive with respect to the contents already
    present in the image.

    Args:

        image (ndarray): The image to render to. The image is expected to real
        valued with values between 0 and 1.

        locations (list of tuple of floats): The locations (in pixels)
        where to place the points.

        sigma (float): The standard deviation of the Gaussian
        point-spread-function.
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

    # blurs the image according to the Gaussian psf
    scipy.ndimage.gaussian_filter(
        point_image,
        sigma,
        output=point_image,
        truncate=3.0)

    # subract point image from original image
    image -= point_image

    # ensure values are still between 0 and 1
    np.clip(image, 0.0, 1.0, out=image)


def render_cage(volume, location, cage, sigma):
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

    for zplane in range(0, depth):
        valid_locs = cage_point_locs[(cage_point_locs[:, 0] >= zplane) &
                                     (cage_point_locs[:, 0] < (zplane + 1)) &
                                     (cage_point_locs[:, 1] >= 0) &
                                     (cage_point_locs[:, 1] < height) &
                                     (cage_point_locs[:, 2] >= 0) &
                                     (cage_point_locs[:, 2] < width)]
        render_points(volume.resolution,
                      volume.data[zplane, :, :],
                      valid_locs[:, 1:3],
                      sigma)


def render_cage_distribution(volume, cage, sigma, density, mask=None):
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
    # TODO: should be in points per micron
    num_expected_points = int(density*(depth*height*width))

    locations = (
        np.random.random((num_expected_points, 3)) *
        [depth, height, width]
    )

    # filter locations by mask (if given)
    if(mask is not None):
        voxel_locations = locations/mask.resolution
        voxel_locations = voxel_locations.astype(np.int32)
        mask_depth, mask_height, mask_width = mask.data.shape
        voxel_locations[:, 0] = np.clip(voxel_locations[:, 0], 0, mask_depth - 1)
        voxel_locations[:, 1] = np.clip(voxel_locations[:, 1], 0, mask_height - 1)
        voxel_locations[:, 2] = np.clip(voxel_locations[:, 2], 0, mask_width - 1)

        indices = (
            voxel_locations[:, 0]*mask_height*mask_width +
            voxel_locations[:, 1]*mask_width +
            voxel_locations[:, 2])
        valid = mask.flatten()[indices].astype(np.bool)
        locations = locations[valid]

    # for each location:
    for loc in locations:

        # randomly rotate cage (account for gimbal lock)
        cage.set_random_rotation()

        # render cage
        render_cage(volume, loc, cage, sigma)


def simulate_cages(volume, segmentation, cages, densities, sigma):
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
        render_cage_distribution(volume, cage, sigma, density, mask)
