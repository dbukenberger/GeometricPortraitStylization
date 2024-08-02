from util import *
from SampleTree import *
from Palette import *
from TwoColoredPixel import *
from SuperPixel import *


class SuperPixelRelaxer:

    def __init__(self, imFile, scale = 1, nSPs = 64, dMap = False, k = 0, nCols = 8, skipJaw = False, colorMode = False):
        np.random.seed(23)      

        self.imFile = imFile        
        self.beta = 1.1
        self.colorMode = colorMode

        iMapFile = self.imFile.split('.')[0]+'Im.png'
        if os.path.exists(iMapFile):
            print('iMap loaded')
            self.importanceMap = cv2.imread(iMapFile, cv2.IMREAD_GRAYSCALE)
        else:
            self.importanceMap = None

        print('Input:', self.imFile)
        imIn = imResize(cv2.imread(self.imFile), scale)
        self.imIn = np.float32(imIn)/255
        self.imLab = cv2.cvtColor(self.imIn, cv2.COLOR_BGR2LAB)

        self.numPixels = np.prod(self.imIn.shape[:2])
        self.dims = np.int32(np.float32(self.imIn.shape[:2])/np.round(np.sqrt(self.numPixels/nSPs)))
        print('BaseDims:', self.dims)
        self.setImDims()
        S = np.float32(self.imIn.shape[:2])/self.dims

        coords = np.reshape(np.dstack(np.meshgrid(range(self.dims[1]),range(self.dims[0]))[::-1]),(-1,2))
        centers = np.int32(coords*S + S/2)
        self.centers = centers
        #self.centers[:,:2] += np.random.randint(-8,9, (self.centers.shape[0],2))

        self.numDetailLevels = k
        if type(dMap) == 'str': # load dMap from file
            dIm = imResize(cv2.imread(dMap, cv2.IMREAD_GRAYSCALE), self.imIn.shape[:2][::-1])
            self.detailLevels = np.int32(np.round(np.float32(dIm)/255*k))
        elif dMap is None:      # noise for max depth everywhere
            dIm = normZeroToOne(np.random.rand(self.imShape[0], self.imShape[1]))
            self.detailLevels = np.int32(np.round(dIm*k))
        else:                   # face features (+ saliency map)
            dIm = drawFaceLandmarks(np.uint8(self.imIn*255), 0, skipJaw)
            if dMap:
                self.detailLevels = computeDetailLevels(self.imIn, k, dIm/255)
            else:
                self.detailLevels = np.int32(np.round(np.float32(dIm)/255*k))
            
        self.s = SampleTree(centers, S, self.detailLevels)
        self.centers = self.s.getSpData()

        r = 2/np.sqrt(len(self.centers)/np.float32(self.numPixels))
        lapSmooth = 0.25
        self.superPixels  = [SuperPixel(self.imLab, c[:-2], i, c[-2:], r, lapSmooth) for i,c in enumerate(self.centers)]
        
        self.numSuperPixels = len(self.superPixels)
        print('nSPs:', self.numSuperPixels)

        self.palette = Palette(self.getLabCols(), kMax = nCols, Tfinal = 1, alpha = 0.85)
        self.cols = np.uint8(255*np.random.rand(self.numSuperPixels,3))

        self.spIds = np.zeros(self.imIn.shape[0] * self.imIn.shape[1], np.int32)
        self.spDsts = np.zeros(self.imIn.shape[0] * self.imIn.shape[1], np.float32)

    def setImDims(self, crop = True):
        d = np.int32(np.min(np.float32(self.imIn.shape[:2])/self.dims))
        d -= d%2
        self.imShape = self.dims*d
        o = (self.imIn.shape[:2]-self.imShape)//2
        self.numPixels = np.prod(self.imShape)
        
        if crop and o[0] > 0:
            self.imIn = self.imIn[o[0]:self.imShape[0]+o[0],:]
            self.imLab = self.imLab[o[0]:self.imShape[0]+o[0],:]
            self.importanceMap = self.importanceMap[o[0]:self.imShape[0]+o[0],:] if self.importanceMap is not None else None
        if crop and o[1] > 0:
            self.imIn = self.imIn[:,o[1]:self.imShape[1]+o[1]]
            self.imLab = self.imLab[:,o[1]:self.imShape[1]+o[1]]
            self.importanceMap = self.importanceMap[:,o[1]:self.imShape[1]+o[1]] if self.importanceMap is not None else None
        
        d2 = np.power(2, np.int32(np.round(np.log2(d))))
        if d2 != d:
            self.imShape = self.dims * d2
            self.imIn = imResize(self.imIn, tuple(self.imShape[::-1]))
            self.imLab = imResize(self.imLab, tuple(self.imShape[::-1]))
            if self.importanceMap is not None:
                self.importanceMap = imResize(self.importanceMap, tuple(self.imShape[::-1]))
            self.numPixels = np.prod(self.imShape)

    def assignPixels(self):
        self.spIds[:] = -1
        self.spDsts[:] = self.numPixels
        for sp in self.superPixels:
            sp.assignPixels(self.spIds, self.spDsts, self.imLab)

    def recenterSuperpixels(self):
        for sp in self.superPixels:
            sp.recenter(self.spIds, self.imLab)

    def relax(self, preview = None):
        i = 0
        totalTime = 0

        pbar = tqdm(None, total=self.palette.kMax, ascii = True, desc='colors')
        while not self.palette.converged:

            startTime = time()
            self.assignPixels()
            self.recenterSuperpixels()

            self.palette.associateAndRefine(self.getLabCols() if self.numDetailLevels else self.getBilateralColors(), self.getImportanceWeights())
            self.palette.expand(self.superPixels)
            pbar.update(self.palette.k - pbar.n)

            deltaTime = time() - startTime
            totalTime += deltaTime
            if preview is None:
                continue
            else:
                cols = np.float32([sp.lab for sp in self.superPixels])
                im = np.uint8(cv2.cvtColor(cols[self.spIds].reshape(self.imShape[0],self.imShape[1],-1), cv2.COLOR_LAB2BGR) * 255)

                pts = [(tuple(np.int32(sp.yx[::-1])), tuple(np.int32(sp.yxInit[::-1]))) for sp in self.superPixels]
                for sp in self.superPixels:
                    pts = np.int32([sp.yx, sp.yxInit])
                    im = cv2.line(im, pts[0][::-1], pts[1][::-1], [0,0,255], 1, 8)
                    im[pts[:,0], pts[:,1]] = 255
            
                pal = np.uint8(255 * imResize(self.palette.getRgbIm(), (im.shape[1], 64), cv2.INTER_NEAREST))
                if preview == 'window':
                    cv2.imshow('ids', np.concatenate([im,pal],axis=0))
                    if cv2.waitKey(100) % 0x100 == 27:
                        break
                elif preview == 'file':
                    cv2.imwrite('preview.png', np.concatenate([im,pal],axis=0))
            i+=1
        pbar.close()
        print('t:', totalTime, i)

        cv2.destroyAllWindows()

        for sp in self.superPixels:
            sp.pId = np.argmin(norm(self.palette.ys - sp.lab))

    def getImportanceWeights(self):
        if self.importanceMap is not None:
            iWeights = np.ones(self.numSuperPixels)
            for i, sp in enumerate(self.superPixels):
                iWeights[i] = self.importanceMap[sp.coords[:,0],sp.coords[:,1]].mean() if sp.weight else 0
                #iWeights[i] = self.importanceMap[sp.coords.T.tolist()].mean() if sp.weight else 0
        else:
            iWeights = [sp.weight for sp in self.superPixels]
        return np.float32(iWeights)/np.sum(iWeights) if np.sum(iWeights) > 0 else np.float32(iWeights+1)/self.numSuperPixels
        
    def getBilateralColors(self):
        return np.reshape(cv2.bilateralFilter(np.float32(self.getLabIm()), 7, 16, -1), (-1, 3))

    def getLabCols(self):
        return np.array([sp.lab for sp in self.superPixels])

    def getLabIm(self):
        return np.reshape(self.getLabCols(), (self.dims[0],self.dims[1],3))

    def getPixelatedIm(self, style = 'pixels'):
        im = np.zeros_like(self.imIn)
        if style == 'tcp':
            cols = self.palette.getRgbIm(self.beta).reshape(-1,3) if len(self.palette.pys)>1 else None
        for sp in self.superPixels:
            pt = sp.yxInit
            corners = np.int32([np.maximum(pt-sp.S,[0,0]), np.minimum(pt+sp.S, im.shape[:2])])
            if style == 'tcp':
                tile = self.imIn[corners[0,0]:corners[1,0],corners[0,1]:corners[1,1]]
                sp.tcp = TwoColoredPixel(tile, cols)
                tcpRes = sp.tcp.renderBestLine()
                im[corners[0,0]:corners[1,0],corners[0,1]:corners[1,1]] = tcpRes
            else:
                im[corners[0,0]:corners[1,0],corners[0,1]:corners[1,1]] = sp.lab * [1,self.beta,self.beta]
                if style == 'blocks':
                    cv2.rectangle(im, tuple(corners[0,:][::-1]), tuple(corners[1,:][::-1]), [0,0,0] , 1, 1)                
        if style in ['pixels', 'blocks']:
            cv2.cvtColor(im, cv2.COLOR_LAB2BGR, im)
        if style == 'pixels':
            minS = np.min([sp.S for sp in self.superPixels], axis=0)*2
            im = imResize(im, tuple(np.int32(np.round(im.shape[:2]/minS))[::-1]), cv2.INTER_NEAREST)
            
        return boostColors(im) if self.colorMode == 'boost' else im

    def writeTcpSVG(self, svgPath):
        xdom = minidom.parseString('<?xml version="1.0" standalone="yes"?><!DOCTYPE svg><svg version="1.1" viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg" style="shape-rendering:crispEdges;"></svg>'%self.imIn.shape[:2][::-1])

        rgbCols = self.palette.getRgbIm()[0]
        polyShapes, cIds = [], []
        for sp in self.superPixels:
            for polyShape, col in zip(*sp.tcp.getPolyShapes(sp.yxInit-sp.S)):
                polyShapes.append(polyShape)
                cIds.append(np.argmin(norm(rgbCols - col)))

        for pId, colRGB in enumerate(self.colsRGB):
            g = xdom.createElement('g')
            g.setAttribute('style', 'fill:%s'%rgb2hex(colRGB))
            for polyShape, cId in zip(polyShapes, cIds):
                if pId == cId:
                    poly = xdom.createElement('polygon')
                    poly.setAttribute('points', ' '.join(['%g,%g'%tuple(pt[::-1]) for pt in polyShape]))
                    g.appendChild(poly)
            xdom.childNodes[1].appendChild(g)
            
        with open(svgPath, 'w') as fh:
            fh.write(xdom.toprettyxml())

    def writeSVG(self, svgPath):
        xdom = minidom.parseString('<?xml version="1.0" standalone="yes"?><!DOCTYPE svg><svg version="1.1" viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg"></svg>'%self.imIn.shape[:2][::-1])

        for pId, colRGB in enumerate(self.colsRGB):
            g = xdom.createElement('g')
            g.setAttribute('style', 'fill:%s'%rgb2hex(colRGB))
            for sp in self.superPixels:
                if sp.pId == pId:
                    yx = np.int32(np.maximum(sp.yxInit-sp.S,[0,0]))
                    rect = xdom.createElement('rect')
                    rect.setAttribute('y', str(yx[0]))
                    rect.setAttribute('x', str(yx[1]))
                    rect.setAttribute('width', str(int(sp.S[0]*2)))
                    rect.setAttribute('height', str(int(sp.S[0]*2)))
                    g.appendChild(rect)
            xdom.childNodes[1].appendChild(g)

        with open(svgPath, 'w') as fh:
            fh.write(xdom.toprettyxml())

    def saveResults(self, resultDir, fName = None):
        if fName is None:
            fName = os.path.basename(self.imFile).split('.')[0]
        rFilePrefix = resultDir + fName + '_'

        cols = (boostColors(self.palette.getRgbIm(), random = self.colorMode == 'random') if self.colorMode else self.palette.getRgbIm())[0]
        self.colsRGB = np.uint8(255 * cols)[:,::-1]

        qtLevels = self.s.draw(self.detailLevels.copy(), True)
        blues = np.fliplr(np.int32([[32,128,255],[196,224,255]]))
        scale = np.linspace(0, 1, qtLevels.max() + 1)
        qtMap = np.uint8(np.dot(np.transpose([scale,1-scale]), blues))        
        cv2.imwrite(rFilePrefix + 'qth.png', qtMap[qtLevels])

        cv2.imwrite(rFilePrefix + 'src.png', np.uint8(255*np.clip(self.imIn, 0, 1)))
        cv2.imwrite(rFilePrefix + 'dls.png', np.uint8(self.detailLevels / self.detailLevels.max() * 255))
        cv2.imwrite(rFilePrefix + 'res.png', np.uint8(255*self.getPixelatedIm()))
        cv2.imwrite(rFilePrefix + 'tcp.png', np.uint8(255*self.getPixelatedIm('tcp')))
        cv2.imwrite(rFilePrefix + 'sps.png', self.cols[self.spIds.reshape(self.imShape[0], self.imShape[1])])
        self.writeSVG(rFilePrefix + 'res.svg')
        self.writeTcpSVG(rFilePrefix + 'tcp.svg')


if __name__ == '__main__':

    spr = SuperPixelRelaxer('input/hendrix.webp',
                            scale = 0.5,
                            nSPs = 64,
                            dMap = True,
                            k = 3,
                            nCols = 8,
                            skipJaw = False,
                            colorMode = 'boost')
    spr.relax('window')
    spr.saveResults('results/')


