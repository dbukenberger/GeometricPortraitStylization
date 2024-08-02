from util import *

class Palette:

    def __init__(self, labCols, kMax=8, Tfinal = 1, alpha = 0.9):
        self.kMax = kMax
        self.Tfinal = Tfinal
        self.alpha = alpha

        self.k = 1
        self.xs = labCols
        self.ys = np.float32([np.mean(self.xs, axis=0)])
        self.ysOld = []
        self.pys = np.float32([1])
        self.pxs = np.ones(len(self.xs))/len(self.xs)

        self.T = np.var(self.xs ,axis=0).max() * 2.1
        self.converged = False

    def associateAndRefine(self, newCols, iWeights = None):
        self.xs = newCols
        self.pxs = self.pxs if iWeights is None else iWeights
        if len(self.ys) != len(self.ysOld):
            self.ysOld = self.ys*-1
        iters = 0
        while np.max(np.sqrt(np.sum((self.ys-self.ysOld)**2,axis=1))) > 0.1:
            self.ysOld = self.ys.copy()
            # experimental, max or T*3 ?
            e = np.exp(-np.reshape(np.max((np.repeat(self.xs, len(self.ys), 0) - np.tile(self.ys, [len(self.xs), 1]))**2, axis=1), (len(self.xs), len(self.ys))) / self.T)
            #e = np.clip(e, eps, e.max())
            self.pyxs = ((self.pys*e).T/np.dot(e,self.pys)).T
            self.pys = np.dot(self.pxs, self.pyxs)
            self.ys = np.dot(self.pyxs.T*self.pxs,self.xs)/np.transpose([self.pys]*self.ys.shape[1])
            iters += 1
            # cancel oscillation 
            if iters > 100: # or find better convergence criteria
                print(self.T)
                break

    def expand(self, superPixels):
        self.T *= self.alpha
        for i in range(len(self.ys)):
            dists = self.xs-self.ys[i]
            pxy = self.pyxs[:,i]*self.pxs
            eigVals, eigVecs = np.linalg.eig(np.dot(np.multiply(dists.T, pxy), dists))
            if eigVals.max() > self.T and self.k < self.kMax:
                eigVec = eigVecs[:,np.argmax(eigVals)]
                self.ys = np.concatenate([self.ys, [self.ys[i,:]-eigVec]], axis=0)
                self.ys[i] += eigVec
                self.pys = np.concatenate([self.pys, [self.pys[i]*0.5]], axis=0)
                self.pys[i] *= 0.5
                self.k += 1

        for i, sp in enumerate(superPixels):
            dists = np.sqrt(np.sum((self.ys-sp.lab)**2,axis=1))
            sp.lab = self.ys[dists.argmin()]
        self.converged = self.k >= self.kMax and self.T <= self.Tfinal

    def getRgbIm(self, beta = 1):
        return cv2.cvtColor(np.reshape(np.float32(self.ys * [1,beta,beta]), (1,-1,3)),cv2.COLOR_LAB2BGR)
