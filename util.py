try:
    from drbutil import *
except ImportError:
    import sys
    import os
    import requests
    utilDir = 'drbutil/'
    if not os.path.exists(utilDir):
        os.mkdir(utilDir)
        print('drbutil not found, downloading ...')
        for fName in ['__init__', '__version__', 'util', 'io']:
            r = requests.get('https://raw.githubusercontent.com/dbukenberger/drbutil/main/src/drbutil/%s.py'%fName)
            if r.status_code != 200:
                print('Something went wrong, try downloading/installing drbutil manually.')
            else:
                with open(utilDir + fName + '.py', "w+") as pyFile:
                    pyFile.write(r.text)
        print('drbtuil downloaded, now simply rerun the script')
        sys.exit()
else:
    print('drbutil found')

from xml.dom import minidom
import cv2

import dlib
# face predictor model for dlib
predictorPath = 'models/shape_predictor_68_face_landmarks.dat'

if not os.path.exists(predictorPath):
    import requests    
    if not os.path.exists('models'):
        os.mkdir('models')
    print('dlib face model not found, downloading (100 MB) ...')
    dlPath = 'https://huggingface.co/public-data/dlib_face_landmark_model/resolve/main/shape_predictor_68_face_landmarks.dat'
    r = requests.get(dlPath, stream=True)
    if r.status_code != 200:
        print('Something went wrong, try downloading dlib model manually.')
    else:
        dlSize = r.headers.get('content-length', 0)
        pbar = tqdm(None, desc='downloading', total=int(r.headers.get('content-length', 0)), ascii = True, unit = 'B', unit_scale = True, unit_divisor = 1024)
        with open(predictorPath, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1024):
                fh.write(chunk)
                pbar.update(os.path.getsize(predictorPath) - pbar.n)
        pbar.close()
        
landmarkIdxs = {"jaw": (0, 17), "browR": (17, 22), "browL": (22, 27), "nose": (27, 36), "eyeR": (36, 42), "eyeL": (42, 48), "mouth": (48, 68)}

def drawFaceLandmarks(imIn, blur = 5, skipJaw = True):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictorPath)

    imGs = imIn if imIn.ndim == 2 else cv2.cvtColor(imIn, cv2.COLOR_BGR2GRAY)

    rects = detector(imGs, 1)
    lmPtss = []
    for i, rect in enumerate(rects):
        shape = predictor(imGs, rect)
        lmPtss.append(np.int32([[pt.x, pt.y] for pt in shape.parts()]))

    imRes = np.zeros_like(imGs)
    for lmPts in lmPtss:
        for i, name in enumerate(landmarkIdxs.keys()):
            j, k = landmarkIdxs[name]
            pts = lmPts[j:k]
            if name == 'jaw' and skipJaw:
                continue
            if name in ['jaw', 'nose'] or 'brow' in name:
                n = k-j - 1
                segIdxs = np.transpose([np.arange(n), np.arange(n)+1])
                if name == 'nose':
                    segIdxs[3,1] = 6
                for i,j in segIdxs:
                    cv2.line(imRes, pts[i], pts[j], 255, 2)
            else:
                hull = cv2.convexHull(pts)
                cv2.drawContours(imRes, [hull], -1, 255, 2)

    for i in range(blur):
        dElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3), (1,1))
        imRes = cv2.dilate(imRes, dElement)
        imRes = cv2.GaussianBlur(imRes, (5, 5), cv2.BORDER_DEFAULT)
        imRes = np.uint8(imRes * 255.0/imRes.max())
    
    return imRes

def normLab(val):
    return (val+[0, 86.1813, 107.862])/[100, 86.1813+98.2352, 107.862+94.4758]

def imResize(im, s, interp = cv2.INTER_LANCZOS4):
    dSize = s if type(s) == tuple else tuple(np.int32(np.round(np.array(im.shape[:2])[::-1]*s)))
    return cv2.resize(im, dSize, interpolation = interp)

def boostColors(im, H = 1, L = 1.25, S = 1.5, random = False):
    colsHLS = cv2.cvtColor(im, cv2.COLOR_RGB2HLS)
    if random:
        s = np.random.get_state()
        n = len(colsHLS[0])
        colsHLS[0,:,0] = np.random.rand(n) * 360
        colsHLS[0,:,2] = np.random.rand(n)
        np.random.set_state(s)
    return cv2.cvtColor(np.float32(np.clip(colsHLS * [H, L, S], [0,0,0], [360, 1, 1])), cv2.COLOR_HLS2RGB)

def filterSobel2D(im):
    kernel = np.float32([[-1,0,1],[-2,0,2],[-1,0,1]])
    yFilt = cv2.filter2D(im, 5, kernel) # cv2.Sobel(im, 5, 1, 0, 3)
    xFilt = cv2.filter2D(im, 5, kernel.T) #cv2.Sobel(im, 5, 0, 1, 3)
    return normZeroToOne(np.sqrt(np.sum(yFilt**2 + xFilt**2, axis=2)))

def computeDetailLevels(imIn, k, dMap = None):
    gPyr = [imIn]
        
    while min(gPyr[-1].shape[:2]) > min(gPyr[0].shape[:2])*0.1:
        lvl = imResize(gPyr[-1], 0.5, cv2.INTER_LINEAR)
        gPyr.append(lvl)

    for i, lvl in enumerate(gPyr[::-1]):
        im = filterSobel2D(lvl) + (imResize(im, 2, cv2.INTER_LINEAR) if i else 0)
    rPyr = normZeroToOne(im)

    if dMap is not None:
        if dMap.max():
            rPyr *= (k-1)/k
        rPyr = np.clip(rPyr + dMap, 0, 1)
                
    return np.uint8(np.round(rPyr * k))
