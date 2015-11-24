import matplotlib.pyplot as plt

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
