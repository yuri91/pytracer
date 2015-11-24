import sys
import math
import numpy as np
import matplotlib.pyplot as plt

from pytracer.core import Camera,Scene,Material,Light
from pytracer.objects import Sphere,Plane
from pytracer.textures import PatternTexture,ImgTexture 

if __name__ == '__main__':
    w = int(sys.argv[1])
    h = int(sys.argv[2])
    aa = int(sys.argv[3])
    ratio = float(w)/h
    
    scene = Scene()

    m1 = Material(1.,1.,50,0.2) 
    m2 = Material(1.,1.,50,0.5) 

    t1 = ImgTexture('earth.png',.3)
    t2 = PatternTexture(lambda u,v:  np.array([0.,0.,1.]) if int(u*10)%2==int(v*10)%2 else np.array([1.,1.,0.]))
    t3 = PatternTexture(lambda u,v:np.array([1.,0.,0.]))
    t4 = PatternTexture(lambda u,v:  np.array([0.1,0.1,0.1]) if int(u*2)%2==int(v*2)%2 else np.array([0.3,0.,0.]))
    
    scene.objects = [
            Sphere([ 0., 0., -4], .5, m1,t1),
            Sphere([-1., 0., -6], .5, m1,t2),
            Sphere([ 1., 0., -2], .5, m1,t3),
            Plane([0., -.5, 0.], [0., 1., 0.],m2,t4,[1.,0.,0.]),
    ]
    scene.lights = [
            Light([5., 5., -10.],[1.,1.,1.]),
            Light([-5., 5., 10.],[1.,1.,1.]),
    ]
    scene.camera = Camera([0.,3.,2.],[0., -0.5, -1],[0.,1,-0.5],math.pi/4,ratio)
    scene.ambient = 0.05

    img = scene.draw(w,h,4,aa)
    plt.imsave('fig.png', img)

