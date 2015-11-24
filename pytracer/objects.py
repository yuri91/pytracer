import math
import numpy as np

from .math_utils import *

class Sphere:
    def __init__(self,center,radius,material,texture):
        self.center = np.array(center)
        self.radius = np.array(radius)
        self.material = material
        self.texture = texture

    def intersect(self,ray):
        a = 1
        CO = ray.origin - self.center
        b = 2 * np.dot(ray.direction, CO)
        c = np.dot(CO, CO) - self.radius * self.radius
        disc = b * b - 4 * a * c
        if disc > 0:
            discSqrt = np.sqrt(disc)
            t0 = (-b - discSqrt)/2.0
            t1 = (-b + discSqrt)/2.0
            t0, t1 = min(t0, t1), max(t0, t1)
            if t1 >= 0:
                return t1 if t0 < 0 else t0
        return np.inf

    def normal(self,point):
        return normalize(point-self.center)

    def get_color(self,point):
        d = normalize(point - self.center)
        u = 0.5 - math.atan2(d[0],d[2])/(math.pi*2) 
        v = 0.5 - math.asin(d[1])/math.pi
        
        return self.texture.get_color(u,v)

class Plane:
    def __init__(self,point,normal,material,texture,texture_x_axis):
        self.point = np.array(point)
        self._normal = normalize(np.array(normal))
        self.material = material
        self.texture = texture
        self.texture_x_axis = texture_x_axis
        self.texture_y_axis = cross(self.texture_x_axis,self._normal)

    def intersect(self,ray):
        denom = np.dot(ray.direction,self._normal)
        if denom == 0:
            return np.inf

        d = np.dot(self.point-ray.origin,self._normal)/denom

        if d<0:
            return np.inf
        return d

    def normal(self,point):
        return normalize(self._normal)

    def get_color(self,point): 
        x = np.dot(self.texture_x_axis,point)
        y = np.dot(self.texture_y_axis,point)

        u = x % 1
        v = y % 1
        
        return self.texture.get_color(u,v)
