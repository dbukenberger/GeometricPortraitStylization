from util import *

class SampleTree:

    offsetDirs = np.float32([[-1,-1],[1,-1],[-1,1],[1,1]])

    def __init__(self, centers, S, sMap, l = 0):
        self.pos = np.int32(np.mean(centers,axis=0))
        self.S = S
        self.l = l
        self.Ss = self.offsetDirs * S
        self.children = []
        if len(centers) == 1:
            self.split(sMap)
        else:
            for c in centers:
                self.children.append(SampleTree([c], S, sMap, self.l+1))
                self.children[-1].split(sMap)

    def split(self, sMap):
        coords = np.int32(np.round(self.pos + self.Ss/2))
        if np.any(sMap[coords[0,0]:coords[-1,0], coords[0,1]:coords[-1,1]] >= self.l):
            offsets = np.round(self.offsetDirs * self.S/4)
            self.children = [SampleTree([self.pos+o], np.ceil(self.S/2), sMap, self.l+1) for o in offsets]

    def getSpData(self):
        if self.children:
            return np.concatenate([c.getSpData() for c in self.children], axis=0)
        else:
            return [np.int32(np.concatenate([self.pos, self.S]))]

    def draw(self, im, lvls = False):
        if self.children:
            for c in self.children:
                im = c.draw(im, lvls)
        else:
            col = np.zeros(3)
            col[self.l%3] = 255
            minMaxPos = lambda i: tuple(np.int32(np.round(self.pos + self.Ss[i]/2)+i))[::-1]
            if lvls:
                cv2.rectangle(im, minMaxPos(0), minMaxPos(-1), self.l, -1, 8)
            else:
                cv2.rectangle(im, minMaxPos(0), minMaxPos(-1), col, 1, 8)
                im[self.pos[0], self.pos[1]] = col
        return im
