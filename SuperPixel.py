from util import *

class SuperPixel:

    def __init__(self, imLab, center, spId, S, r = 1, lapSmooth = 0):
        self.id = spId
        self.yxInit = center
        self.yx = center
        self.S = S/2

        self.yxCoords = np.dstack(np.meshgrid(range(imLab.shape[1]), range(imLab.shape[0])))
        self.yxCoords = np.vstack(self.yxCoords[:,:,::-1])
        
        self.labInit = imLab[center[0],center[1]]
        self.lab = self.labInit.copy()
        self.rScale = r
        self.lapSmooth = lapSmooth
        self.weight = np.prod(S)
        self.distNorm = np.max(self.S*2)

    def assignPixels(self, idMap, dstMap, imLab):
        self.yLims = np.int32(np.round([max(0, self.yx[0]-self.S[0]*2), min(imLab.shape[0], self.yx[0]+self.S[0]*2)]))
        self.xLims = np.int32(np.round([max(0, self.yx[1]-self.S[1]*2), min(imLab.shape[1], self.yx[1]+self.S[1]*2)]))
        self.coords = np.transpose([np.tile(range(self.yLims[0], self.yLims[1]), self.xLims[1]-self.xLims[0]), np.repeat(range(self.xLims[0], self.xLims[1]), self.yLims[1]-self.yLims[0])])
        self.idxs = self.coords[:,0] * imLab.shape[1] + self.coords[:,1]
        
        #self.dyx = np.sqrt(np.sum((self.yx - self.coords)**2,axis=1)) # L_2 euclidean
        #self.dyx = np.sum(np.abs(self.yx - self.coords),axis=1) # L_1 manhattan
        self.dyx = np.max(np.abs(self.yx - self.coords), axis=1) # L_\infty chebyshev
        self.dlab = norm(self.lab - imLab[self.coords[:,0], self.coords[:,1]])
        ds = self.dlab + (self.rScale / self.distNorm) * self.dyx
        
        m = ds < dstMap[self.idxs]
        idMap[self.idxs[m]] = self.id
        dstMap[self.idxs[m]] = ds[m]

    def recenter(self, idMap, imLab):
        msk = idMap == self.id
        if np.any(msk):
            crds = self.yxCoords[msk]
            self.lab = imLab[crds[:,0], crds[:,1]].mean(axis=0)
            self.weight = len(crds)
            self.labDist = norm(normLab(self.lab) - normLab(self.labInit))
            alpha = self.labDist if self.lapSmooth < 0 else self.lapSmooth
            self.yx = (1-alpha) * crds.mean(axis=0) + alpha * self.yxInit
