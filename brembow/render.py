import numpy as np
import scipy
import math

def render_point(image, location, sigma):
    '''Render a single point with a Gaussian point-spread-function into a 2D
    image. This rendering is subtractive with respect to the contents already
    present in the image.

    Args:

        image (ndarray): The image to render to. The image is expected to real
        valued with values between 0 and 1.

        location (tuple of floats): The location (in pixels) where to place the
        point.

        sigma (float): The standard deviation of the Gaussian
        point-spread-function.
    '''
    # now between -1 and 1 to account for modified images from render_cage
    assert(image.min() >= 0 and image.max() <= 1)

    point_image = np.zeros_like(image)

    x, y = location[0], location[1]  # takes x and y from location
    point_image[x, y] = 1.0

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

        volume (ndarray): The volume to render to. The volume is expected to be
        real valued with values between 0 and 1.

        location (tuple of floats): The location (in pixels) where to place the
        cage.

        cage (`class:Cage`): The cage to render.

        sigma (float): The standard deviation of the Gaussian
        point-spread-function.
    '''

    # normalizing points in relation to cage loc
    pre_norm_cage_loc = cage.get_locations()
    cage_point_locations = pre_norm_cage_loc + location

    depth, height, width = volume.shape

    # cycle through each z-plane
    for zplane in range(0, depth):
        for possible_point in cage_point_locations:
            # if s <= z-value < s+1 and point is in-bound of the volume, render
            if (zplane <= possible_point[0] < (zplane + 1) and
                    0 <= possible_point[1] < height and
                    0 <= possible_point[2] < width):
                render_point(
                    volume[zplane, :, :],
                    [possible_point[1], possible_point[2]],
                    sigma)

def render_cage_distribution(volume, cage, sigma, density, mask=None):
    '''Renders randomly oriented copies of the given cage with the given
    density into volume.

    Args:

        volume (ndarray): The volume to render to. The volume is expected to be
        real valued with values between 0 and 1.

        cage (`class:Cage`): The cage to render.

        sigma (float): The standard deviation of the Gaussian
        point-spread-function.

        density (float): The density of cages in number of cages per cubic
        micron.

        mask (ndarray, optional): If given, limit rendering to areas where mask
        is > 0.
    '''
    depth, height, width = volume.shape
    num_expected_points = density*volume
    locations = []
    for x in range (0, num_expected_points):
        locations.append([random.randint(0, depth),random.randint(0, height),random.randint(0, width)])
    for l in locations:
        if(mask[l[0]][l[1]][l[2]] == 0):
            locations.pop(locations.index(l))
    for l in locations:
        cage.set_rotation(random.random()*2*pi,random.random()*2*pi, random.random()*2*pi)
        render_cage(volume, l, cage, sigma)


    # randomly sample locations within the volume's size


    # filter locations by mask (if given)


    # for each location:

        # randomly rotate cage (account for gimbal lock)

        # render cage
    pass

# TODO: change above functions to take arguments in world units

