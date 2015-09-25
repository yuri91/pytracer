import math
import numpy as np
import matplotlib.pyplot as plt


def normalize(x):
    x /= np.linalg.norm(x)
    return x

def cross(x,y):
    i = x[1]*y[2]-x[2]*y[1]
    j = x[2]*y[0]-x[0]*y[2]
    k = x[0]*y[1]-x[1]*y[0]
    
    v = np.array([i,j,k])

    return v

class Ray:
    def __init__(self,origin,direction):
        self.origin = np.array(origin)
        self.direction = normalize(np.array(direction))

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




class Camera:
    def __init__(self,origin,direction,up_dir,fov,ratio):
        self.origin = np.array(origin)
        self.direction = normalize(np.array(direction))
        self.up_dir = normalize(np.array(up_dir))
        self.fov = fov
        self.ratio=ratio
    
    def ray_generator(self,resolution_x,resolution_y):
        def gen():
            fov_x = self.fov
            fov_y = self.fov/self.ratio

            base_dir = self.direction
            y_dir = self.up_dir
            x_dir = cross(base_dir,y_dir)

            orig = self.origin

            for i in range(resolution_x):
                x = i-resolution_x/2
                alpha_x = x*fov_x/resolution_x
                dx = math.tan(alpha_x)
                for j in range(resolution_y):
                    y = j-resolution_y/2
                    alpha_y = y*fov_y/resolution_y
                    dy = math.tan(alpha_y)

                    dv = y_dir*dy+x_dir*dx
                    d = normalize(base_dir+dv)

                    yield i,j,Ray(orig,d)

        return gen


class Light:
    def __init__(self,origin,color):
        self.origin = np.array(origin)
        self.color = np.array(color)


class Material:
    def __init__(self,diffuse_c,specular_c,specular_k,reflection):
        self.diffuse_c = diffuse_c
        self.specular_c = specular_c
        self.specular_k = specular_k
        self.reflection = reflection

class PatternTexture:
    def __init__(self,uv_func):
        self._uv_func = uv_func

    def get_color(self,u,v):
        return self._uv_func(u,v)

class ImgTexture:
    def __init__(self,f,offset=0):
        self.img = plt.imread(f)
        self.w = self.img.shape[0]
        self.h = self.img.shape[1]
        self.offset = offset

    def get_color(self,u,v):
        u += self.offset
        u = u if u<1 else u-1 
        return self.img[int(u*self.w),int(v*self.h)]


class Scene:
    def __init__(self):
        self.objects = []
        self.lights = []
        self.camera = Camera([0.,0.,0.],[0., 0., -1.],[0.,1.,0.],math.pi/3,1)
        self.ambient = 0.


    def intersect(self,ray):
        # Find first point of intersection with the scene.
        t_min = np.inf
        obj_min = None
        for obj in self.objects:
            t_obj = obj.intersect(ray)
            if t_obj < t_min:
                t_min, obj_min = t_obj, obj
        # Return None if the ray does not intersect any object.
        if t_min == np.inf:
            return None
        # Find the point of intersection on the object.
        P = ray.origin + ray.direction * t_min
        #return object intersected and point of intersection
        return (obj_min,P)

    def trace_ray(self,ray):
        intersection = self.intersect(ray)
        if intersection == None:
            return None
        obj,P = intersection
        N = obj.normal(P)
        cameraDir = normalize(self.camera.origin- P) 

        # ambient light
        color = self.ambient
        
        for light in self.lights:
            lightDir = normalize(light.origin - P)
            lightIntersection = self.intersect(Ray(P+0.0001*N,lightDir))
            if lightIntersection == None:
                # Lambert shading (diffuse).
                color += obj.material.diffuse_c*max(np.dot(N,lightDir),0) * obj.get_color(P)
                # Blinn-Phong shading (specular).
                color += obj.material.specular_c * max(np.dot(N, normalize(lightDir + cameraDir)), 0) ** obj.material.specular_k * light.color

        return color,P,N,obj

        

    def draw(self,w,h,depth,antialias):
        W = w*antialias
        H = h*antialias
        r = float(W) / H
        color = np.zeros(3)
        IMG = np.zeros((H,W,3))
         
        rays = self.camera.ray_generator(W,H)
        # Loop through all pixels.
        oldperc = -1
        for i, j, ray in rays():
            perc = int((i*H+j)*100/(W*H))
            if perc % 5  == 0 and perc != oldperc:
                print (perc, "%")
                oldperc = perc
            color = np.zeros(3)
            reflection = 1.


            for k in range(depth):
                traced = self.trace_ray(ray)
                if traced == None:
                    break
                c,P,N,obj = traced
                color += reflection*c
                reflection *= obj.material.reflection
                
                #reflected ray
                O = P+N*0.0001
                D = normalize(ray.direction - 2*np.dot(ray.direction,N)*N)
                ray = Ray(O, D)

            IMG[H - j - 1, i, :] = np.clip(color, 0, 1)
         
        img = np.zeros((h,w,3))
        for i in range(h):
            for j in range(w):
                color = np.zeros(3)
                for k in range(antialias):
                    for l in range(antialias):
                        color += IMG[i*antialias+k,j*antialias+l]
                img[i,j] = color/(antialias**2)
        return img

if __name__ == '__main__':
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
    scene.camera = Camera([0.,3.,2.],[0., -0.5, -1],[0.,1,-0.5],math.pi/4,4./3.)
    scene.ambient = 0.05

    img = scene.draw(800,600,4,1)
    plt.imsave('fig.png', img)
