import numpy as np
from math import sin, cos
from pyquaternion import Quaternion

class Cage:
    # list of locations in 3D
    def __init__(self, locations):

        self.locations = np.array(locations, np.float32)

        self.theta_z = 0.0
        self.theta_y = 0.0
        self.theta_x = 0.0

        self.rotation_changed = False
        self.rotated = self.locations

    def get_locations(self):

        if rotation_changed:

            self.rotated = np.array(self.locations)
            q1 = Quaternion(axis=[1,0,0], angle=self.theta_x)
            q2 = Quaternion(axis=[0,1,0], angle=self.theta_y)
            q3 = Quaternion(axis=[0,0,1], angle=self.theta_z)
            q  = q1*q2*q3
            self.rotated = q.rotate(self.rotated)
            '''if self.theta_z != 0:
                self.rotate_z(self.rotated, self.theta_z)
            if self.theta_y != 0:
                self.rotate_y(self.rotated, self.theta_y)
            if self.theta_x != 0:
                self.rotate_x(self.rotated, self.theta_x)'''

            self.rotation_changed = False

        return self.rotated

    def set_rotation(self, theta_z, theta_y, theta_x):

        self.theta_z = theta_z
        self.theta_y = theta_y
        self.theta_x = theta_x

        self.rotation_changed = True

'''    def rotate_z(self, locations, theta):
        sin_theta = sin(theta)
        cos_theta = cos(theta)

        for n in range(len(locations)):
            node = locations[n]
            x = node[2]
            y = node[1]
            node[2] = x * cos_theta - y * sin_theta
            node[1] = y * cos_theta + x * sin_theta

    def rotate_y(self, locations, theta):
        sin_theta = sin(theta)
        cos_theta = cos(theta)
        for n in range(len(locations)):
            node = locations[n]
            x = node[2]
            z = node[0]
            node[2] = x * cos_theta + z * sin_theta
            node[0] = z * cos_theta - x * sin_theta

    def rotate_x(self, locations, theta):
        sin_theta = sin(theta)
        cos_theta = cos(theta)

        for n in range(len(locations)):
            node = locations[n]
            y = node[1]
            z = node[0]
            node[1] = y * cos_theta - z * sin_theta
            node[0] = z * cos_theta + y * sin_theta '''
