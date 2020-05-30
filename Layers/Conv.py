import numpy as np

class Conv2D:

    def __init__(self, qfilt, override=False, inpK=None):
        self.shape = (3,3)
        self.numfilters = qfilt
        if override:
            self.filts = inpK
        else:
            self.filts = np.random.randn(qfilt, 3, 3)/9
    
    def convolve(self, image):
        self.img = image

        (height, width) = image.shape
        result = np.zeros((self.numfilters, height-2, width-2))
        self.h = height
        self.w = width
        
        for c in range(self.numfilters):
            for i in range(height-2):
                for j in range(width-2):
                        result[c,i,j] = np.sum(image[i:i+3, j:j+3]*self.filts[c,:,:])
        
        return result

    def train(self, grad, alpha):
        filtderiv = np.zeros(self.filts.shape)

        for c in range(self.numfilters):
            for i in range(self.h-2):
                for j in range(self.w-2):
                    filtderiv[c] += grad[c,i,j] * self.img[i:i+3, j:j+3]
        
        self.filts -= alpha*filtderiv