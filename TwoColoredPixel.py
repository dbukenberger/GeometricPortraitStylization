from util import *

def leftOf(pt, ft):
    f, t = ft
    return np.dot([-(t[1]-f[1]),t[0]-f[0]], (pt-f).T if pt.ndim > 1 else (pt-f)) > 0

class TwoColoredPixel:

    def __init__(self, imIn, cols = None):
        self.imIn = imIn
        self.cols = cols

        self.crop = False
        dims = np.int32(self.imIn.shape[:2])
        if np.any(np.log2(dims) == np.round(np.log2(dims))):
            self.imIn = imResize(imIn, tuple([np.int32(np.round(2**np.floor(np.log2(dims[0]))+1))]*2), cv2.INTER_LINEAR)
            self.crop = True

        self.TCP = np.zeros_like(self.imIn)
        self.dims = np.int32(self.imIn.shape[:2])

        mg = np.dstack(np.meshgrid(np.arange(self.dims[0]), np.arange(self.dims[1]))[::-1])
        self.mgCoords = mg.reshape(-1,2)
        coords = np.concatenate([mg[0,:], mg[1:-1,-1]], axis=0)
        self.coords = np.float32(np.concatenate([coords, self.mgCoords.max(axis=0)-coords],axis=0)+0.5)

        basePixels = np.arange(8)*self.coords.shape[0]//8
        connections = []
        for i, bp in enumerate(basePixels):
            for j, op in enumerate(basePixels):
                if j in np.int32([i-1,i,i+1])%8 or (not bp%8 and j in np.int32([i-2,i+2])%8):
                    continue
                connections.append((i,j))

        self.lines = basePixels[unique2d(np.sort(connections, axis=1))]
        self.offsets = np.int32([(u,v) for u in [-1,0,1] for v in [-1,0,1] if u or v])
        self.bestLine = None
    
    def createMask(self, line):
        idxs = leftOf(self.mgCoords, self.coords[line])
        m = np.zeros_like(self.imIn, np.bool_)
        m[self.mgCoords[idxs,0], self.mgCoords[idxs,1]] = 1
        return m

    def updateBestLine(self, line):
        mL = self.createMask(line)[:,:,0]
        mR = self.createMask(line[::-1])[:,:,0]

        if self.cols is not None:
            if not mL.sum():
                mL[:] = 1
            if not mR.sum():
                mR[:] = 1
            cL = self.cols[np.argmin(norm(self.cols-self.imIn[mL].mean(axis=0)))]
            cR = self.cols[np.argmin(norm(self.cols-self.imIn[mR].mean(axis=0)))]
        else:
            cL = self.imIn[mL].mean(axis=0)
            cR = self.imIn[mR].mean(axis=0)
        
        lineEnergy = np.abs(self.imIn[mL]-cL).sum() + np.abs(self.imIn[mR]-cR).sum()
        if self.bestLine is None or self.bestLine[-1] > lineEnergy:
            self.bestLine = [line, np.float32([cL,cR]), lineEnergy]
            
    def findBestLine(self, level = 0):
        if not level:
            for i, line in enumerate(self.lines):
                self.updateBestLine(line)
        else:
            line = self.bestLine[0].copy()
            map(lambda offset: self.updateBestLine(np.sort((line+offset)%self.coords.shape[0])), self.offsets * (self.dims/(2**(level+1))))

        if np.all(2**(level+2) < self.dims):
            self.findBestLine(level+1)

    def renderBestLine(self):
        if self.bestLine is None:
            self.findBestLine()

        cols = self.bestLine[1]
        mL = self.createMask(self.bestLine[0])
        mR = self.createMask(self.bestLine[0][::-1])
        self.TCP = np.float32(np.ones_like(self.TCP)*cols[0]*np.clip(mL-mR*0.5,0,1)+np.ones_like(self.TCP)*cols[1]*np.clip(mR-mL*0.5,0,1))
        self.TCP[np.where(np.sum(mL+mR,axis=2)==0)] = cols.mean(axis=0)

        if self.crop:
            self.TCP = self.TCP[:-1,:-1]
        return self.TCP

    def getEdgeCoords(self, corners, line):
        s, sLen = normVec(line[0]-line[1], True)
        if 0 in s:
            dim = int(s[0] == 0)
            line[:,dim] = [0,self.dims[dim]]
            edgeCoords = line
        elif np.all(np.abs(s) == np.abs(s).mean()) and np.abs(sLen - norm(np.float32(self.dims)-1)) < eps:
            edgeCoords = corners[[1,-2]] if np.any(np.sign(s)>0) else corners[[0,-1]]
        else:
            q = np.mean(line,axis=0)
            ps = corners[[0,0,3,3]]
            rs = normVec(corners[[0,0,3,3]] - corners[[1,2,1,2]])
            us = np.cross(q-ps, rs/np.cross(rs,s).reshape(-1,1))
            edgeCoords = np.float32(np.array(q + s * us[np.abs(us).argsort()[:2]].reshape(2,1)))
        return edgeCoords if np.abs(edgeCoords-self.coords[self.bestLine[0]]).sum() < np.abs(edgeCoords[::-1]-self.coords[self.bestLine[0]]).sum() else edgeCoords[::-1]
            
    def getPolyShapes(self, offset = [0,0]):
        corners = np.float32([[0,0],[0,1],[1,0],[1,1]] * self.dims)

        if np.allclose(self.bestLine[1][0], self.bestLine[1][1]):
            return [corners[[0,1,3,2]] + offset], [self.bestLine[1][0]]
        
        linePoints = self.getEdgeCoords(corners, self.coords[self.bestLine[0]])

        lPts = np.dot(Mr2D(np.pi * 1.25), np.fliplr(self.coords[self.bestLine[0]] - self.dims/2.0).T).T
        linePoints = linePoints[[0,1] if 0 in self.bestLine[0] else np.argsort(np.arctan2(lPts[:,0], -lPts[:,1]))]

        polyShapes = [cv2.convexHull(np.concatenate([linePoints, corners[leftOf(corners, linePoints[::i])]], axis=0)).reshape(-1,2) + offset for i in [1,-1]]
        return polyShapes, [self.bestLine[1][0], [self.bestLine[1][1]]]
