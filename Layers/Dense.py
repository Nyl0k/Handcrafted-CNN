import numpy as np

class Dense10:
    def __init__(self, inpsize, classes):
        self.inplen = inpsize[0]*inpsize[1]*inpsize[2]
        self.weights = np.random.randn(self.inplen, classes) / self.inplen
        self.biases = np.zeros(classes)
    
    def predict(self, inp):
        flatinp = inp.flatten()

        raw = np.dot(flatinp, self.weights) + self.biases

        self.preflatshape = inp.shape
        self.inp = flatinp
        self.raw = raw
        
        return np.exp(raw)/np.sum(np.exp(raw), axis=0)

    def train(self, initgrad, alpha):
        for i, grad in enumerate(initgrad):
            if grad == 0:
                continue
            
            expt = np.exp(self.raw)

            S = np.sum(expt)

            retgrad = -expt[i] * expt / S**2
            retgrad[i] = expt[i] * (S - expt[i]) / (S**2)

            Ltderiv = grad * retgrad

            Lwderiv = self.inp[np.newaxis].T @ Ltderiv[np.newaxis]
            Lbderiv = Ltderiv*1
            Linpderiv = self.weights @ Ltderiv

            self.weights -= alpha * Lwderiv
            self.biases -= alpha*Lbderiv

            return Linpderiv.reshape(self.preflatshape)