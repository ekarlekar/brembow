
import numpy as np
import scipy



def render_point(image, location, sigma):
  '''Render a single point with a Gaussian point-spread-function into a 2D image. This rendering is subtractive with respect to the contents already present in the image.

  Args:

    image (ndarray): The image to render to.

    location (tuple of floats): The location (in pixels) where to place the point.

    sigma (float): The standard deviation of the Gaussian point-spread-function.
  '''
	image = image/255. #normalize the image so the values are between 0 and 1
	image_copy = copy.deepcopy(image) #deep copy
	iterations = 1 #could change this or make this an argument in the future
	ksize = int(3*sigma) if (int(3*sigma)%2==1) else int(3*sigma)+1 #calculates kernel size as 3*sigma
	mask = np.zeros(image.shape[:2]) #creates a mask
	x, y = location[0], location[1] #takes the x and y values from the location argument
	contours = np.array[(x, y)] #creates a contour of that point
	for i, contour in enumerate(contours):
	    cv2.drawContours(mask, contour, 0, 1., -1) #draw contours
	mask_copy = copy.deepcopy(mask) #deep copy
	blurred_image = cv2.GaussianBlur(image, (ksize,ksize), sigma, None, sigma) #blurs the image according to the Gaussian psf
	blurred_copy = copy.deepcopy(blurred_image) #deep copy
	result = np.copy(image)
	for _ in xrange(iterations):
	    mask = cv2.GaussianBlur(mask, (ksize, ksize), sigma, None, sigma) #blurs mask
	    alpha = mask
	#adds a percentage of the blurred image and the complementary percentage of the original image
	#i.e. blurred image will be weighted more in the center of the kernel
	result = alpha[:, :, None]*blurred_image + (1-alpha)[:, :, None]*result 
	blurred_mask_copy = copy.deepcopy(mask)
	result = (result*255).astype(int) #changes the values back to between 0 and 255
	for i in range(0, len(image)):
	    for j in range(0, len(image[0])):
	        image[i][j] = result[i][j] #copies result into the original image


def render_cage(volume, location, cage, sigma):
  # TODO: write documentation
  # TODO: write implementation
  pass
