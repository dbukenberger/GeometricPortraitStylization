from SuperPixelRelaxer import *

if __name__ == '__main__':
    
    inputDir = 'input/'
    resultDir = 'results/'

    preview = 'window' # or 'file' or None

    inputs = []

    # # # usage # # #
    # inputs.append([imFile,, scale, nSPs, k, nCols, dMap, skipJaw, colorMode, rName])
    # using these parameters
    #
    # imFile    located in inputDir
    # scale     factor on the input image size
    # nSPs      number of superpixels as nSPs^2
    # k         number of levels in quadtree
    # nCols     number of colors in palette
    # dMap      use map for detail levels
        # True          automatic saliency map + facial features
        # False         no saliency map, only facial features
        # None          use noise, max detail depth everywhere
        # 'dMap.png'    str path to load dMap from file
    # skipJaw   exlude jawline from facial features
    # colorMode modify colors for export
        # 'random'  for randomized colors
        # 'boost'   exaggerated colors
        # False     no color modification
    # rName     alternative file name for results


    # # # src wiki # # #

    # https://commons.wikimedia.org/wiki/File:JC1970.jpg
    inputs.append(['cash.jpg', 0.75, 16, 3, 5, True, False, False, None])
    
    # https://commons.wikimedia.org/wiki/File:Gustave_Courbet_-_Le_D%C3%A9sesp%C3%A9r%C3%A9_(1843).jpg
    inputs.append(['courbet.jpg', 0.5, 16, 3, 8, True, True, 'boost', None])

    # https://commons.wikimedia.org/wiki/File:Hitchcock,_Alfred_02.jpg
    inputs.append(['hitchcock.jpg', 0.75, 16, 3, 6, True, True, 'random', None])

    # https://commons.wikimedia.org/wiki/File:Official_portrait_of_Barack_Obama.jpg
    inputs.append(['obama.png', 0.25, 8, 2, 8, True, False, False, None])

    # https://commons.wikimedia.org/wiki/File:1665_Girl_with_a_Pearl_Earring.jpg
    inputs.append(['vermeer.jpg', 0.5, 8, 4, 8, True, True, 'boost', None])
    inputs.append(['vermeer.jpg', 0.5, 8, 4, 8, False, True, 'boost', 'vermeeer'])


    # # # src flickr # # #
    
    # https://www.flickr.com/photos/oneeighteen/3372087394/
    inputs.append(['jaipur.png', 1, 8, 3, 8, True, True, False, None])

    # https://www.flickr.com/photos/sjueline/32826100291/
    inputs.append(['sjueline.png', 0.7529, 16, 3, 10, False, False, False, None])


    # # # src other # # #

    # https://www.rollingstone.com/politics/politics-lists/president-obama-on-the-cover-of-rolling-stone-20420/rs1056-1057-221776/
    inputs.append(['barack.png', 1, 16, 3, 10, True, False, False, None])

    # https://www.bundesdruckerei-gmbh.de/files/dokumente/pdf/fotomustertafel.pdf
    inputs.append(['erika.jpg', 1, 16, 2, 16, True, False, False, None])
    inputs.append(['erika.jpg', 1, 16, 2, 32, None, False, False, 'erikaa'])

    # https://lastfm.freetls.fastly.net/i/u/ar0/52bd3bdd54a09ecd1617146d6e401b02.jpg
    inputs.append(['hendrix.webp', 0.5, 8, 3, 8, True, False, 'boost', None])

    # https://www.carredartistes.com/de-de/content_images/Sans%20titre36.png
    inputs.append(['kahlo.png', 0.5, 8, 3, 8, True, True, 'boost', None])

    # https://www.hlevkin.com/hlevkin/06testimages.htm
    inputs.append(['monarch.tif', 1, 8, 3, 8, True, True, 'boost', None])
    inputs.append(['rainier.bmp', 0.75, 8, 3, 8, True, True, 'boost', None])
    

    for imFile, scale, nSPs, k, nCols, dMap, skipJaw, colorMode, rName in inputs:
        spr = SuperPixelRelaxer(inputDir + imFile, scale, nSPs**2, dMap = dMap, k = k, nCols = nCols, skipJaw = skipJaw, colorMode = colorMode)
        spr.relax(preview)
        spr.saveResults(resultDir, rName)
