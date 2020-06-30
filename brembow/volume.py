
class Volume:
    '''A class representing a 3D volume of data.

    data (3D ndarray): The data of the volume.

    resolution (tuple of float): The size of a voxel in Ångström.
    '''

    def __init__(self, data, resolution):
        self.data = data
        self.resolution = resolution
