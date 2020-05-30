import numpy as np

class MaxPool2D:
    def pool(self, fmap):
        self.inp = fmap

        (numLayers, height, width) = fmap.shape
        output = np.zeros((numLayers, height//2, width//2))
        for c in range(numLayers):
            for i in range(height//2):
                for j in range(width//2):
                    output[c,i,j] = np.amax(fmap[c, i*2:i*2+2, j*2:j*2+2])
        
        return output
    
    def train(self, inpgrad):
        Linpderiv = np.zeros(self.inp.shape)
        (numLayers, height, width) = self.inp.shape
        for c in range(numLayers):
            for i in range(height//2):
                for j in range(width//2):
                    for ii in range(i*2, i*2+2):
                        for jj in range(j*2, j*2+2):
                            if self.inp[c, ii, jj] == np.amax(self.inp[c, i*2:i*2+2, j*2:j*2+2]):
                                Linpderiv[c, ii,jj] = inpgrad[c, i, j]
        
        return Linpderiv