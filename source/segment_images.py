import os
import shutil
import sys
import cv2
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from PIL import ImageFilter
from scipy import stats
import datetime

# import rasterio with warnings disabled to suppress the following warning:
# FutureWarning: GDAL-style transforms are deprecated and will not be supported in Rasterio 1.0
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import rasterio


# define legend colours which were extracted by the legend_colours.py module
matureTimberHSV         = np.array([ 41, 167, 143], dtype='uint8')
immatureTimberHSV       = np.array([ 26, 217, 206], dtype='uint8')
notRestockedHSV         = np.array([ 25, 229, 237], dtype='uint8')
nonCommercialStandsHSV  = np.array([ 89,  92, 154], dtype='uint8')
nonForestedLandHSV      = np.array([ 18,  51, 240], dtype='uint8')
waterHSV                = np.array([ 27,  34, 203], dtype='uint8')
boundaryHSV             = np.array([ 16,  52,  77], dtype='uint8')

matureTimberRGB         = np.array([107, 143,  49], dtype='uint8')
immatureTimberRGB       = np.array([206, 187,  30], dtype='uint8')
notRestockedRGB         = np.array([237, 202,  23], dtype='uint8')
nonCommercialStandsRGB  = np.array([ 98, 153, 151], dtype='uint8')
nonForestedLandRGB      = np.array([240, 221, 191], dtype='uint8')
waterRGB                = np.array([203, 200, 175], dtype='uint8')
boundaryRGB             = np.array([ 77,  70,  61], dtype='uint8')

matureTimberGray        = 99
immatureTimberGray      = 141
notRestockedGray        = 154
nonCommercialStandsGray = 134
nonForestedLandGray     = 217
waterGray               = 192
boundaryGray            = 69

matureTimberStr         = "mature timber"
immatureTimberStr       = "immature timber"
notRestockedStr         = "not restocked"
nonCommercialStandsStr  = "non-commercial stands"
nonForestedLandStr      = "non-forested land"
waterStr                = "water"
boundaryStr             = "boundary"



################################################################################

# Use rasterio.open() to read in data from the selected geoTIFF file and return
# an RGB image and profile with geodata

def readGeoTiff(imageFilePath):

    # read in RGB data from file
    dataset = rasterio.open(imageFilePath)
    imageR = dataset.read(1)
    imageG = dataset.read(2)
    imageB = dataset.read(3)

    # save the data and coordinates of the imported file
    profile = dataset.profile

    # extract the dimensions of the image to determine the number of rows and
    # in the image RGB channels.
    shape = imageR.shape

    # create variables to store row, col, and channel sizes
    numRows = shape[0]
    numColumns = shape[1]
    numChannels = 3

    # create empty black image (np.zeros makes it black)
    imageRGB = np.zeros((numRows,numColumns,numChannels), dtype='uint8')

    # replace the empty image channels with the imported RGB channels to create
    # a single RGB image
    imageRGB[:,:,0] = imageR
    imageRGB[:,:,1] = imageG
    imageRGB[:,:,2] = imageB

    return imageRGB,profile

################################################################################

# save image to disk as PNG using PILLOW library
def saveImagePng(imageArray,destinationFolder,fileName):
    # create new output directory if it does not yet exist
    if not os.path.exists(destinationFolder):
        os.makedirs(destinationFolder)

    imageFormat = "png"
    savePath = destinationFolder + os.path.sep + fileName + "." + imageFormat
    imageObject = Image.fromarray(imageArray)
    imageObject.save(savePath,format=imageFormat)

################################################################################

def saveProcessingStep(imageArray,destinationFolder,fileName):

    # create local counter to increment each time the function is called
    # see https://stackoverflow.com/questions/279561/what-is-the-python-equivalent-of-static-variables-inside-a-function
    if not hasattr(saveProcessingStep, "counter"):
        saveProcessingStep.counter = 0  # it doesn't exist yet, so initialize it
    saveProcessingStep.counter += 1

    # convert counter to string with preceding 0 for single digits
    numString = str(saveProcessingStep.counter)
    if saveProcessingStep.counter < 10:
        numString = str(0) + numString

    stepName = numString + " " + fileName
    saveImagePng(imageArray,destinationFolder,stepName)

################################################################################

# save image to disk using Rasterio.open() from Rasterio
def saveImageTiff(imageArray,profile,destinationFolder,fileName):
    # create new output directory if it does not yet exist
    if not os.path.exists(destinationFolder):
        os.makedirs(destinationFolder)

    # rearrange the array into a format dst.write will like and not get confused with a value error
    tiffArray = np.moveaxis(imageArray,-1,0)

    # determine shape of output image
    shape = imageArray.shape
    height = int(shape[0])
    width = int(shape[1])

    # open a gdal enviroment to use with rasterio
    with rasterio.Env():

    # modify the width and height to the newly cropped image so no errors will occur
        profile.update(
            width=width,
            height=height,
            dtype=rasterio.ubyte,
            count=3,
            compress='lzw')

    # create a new file with the specific name
        savePath = destinationFolder + os.path.sep + fileName + ".tiff"
        with rasterio.open(savePath,'w',**profile) as dst:
            dst.write(tiffArray)

################################################################################

def segmentImage(imageArray,k):
    imageShape = imageArray.shape
    numRows = imageShape[0]
    numColumns = imageShape[1]

    # reshape image so that it has three column 32bit float features as required by the kmeans function
    imageFeatures = np.float32(imageArray.reshape((-1,3)))

    # set algorithm stopping criteria (based on OpenCV k-means function documentation)
    # this controls what happens if max iterations reached or if cluster assignments do not change (small epsilon)
    maxIterations = 10
    epsilon = 1
    terminationType = cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS
    criteria = (terminationType, maxIterations, epsilon)
    attempts = 10
    flags = cv2.KMEANS_RANDOM_CENTERS # specify how initial centers are taken


    # segment the image with k-means and extract data about the labels and centers
    compactness, labels, centers = cv2.kmeans(imageFeatures, k, None, criteria, attempts, flags)

    centers = np.uint8(centers)

    labels = labels.flatten()
    labelsImage = np.copy(labels)
    labelsImage = labelsImage.reshape((numRows,numColumns))
    return labelsImage, centers

################################################################################

# reconstruct segmented image using labels and centers
def createRGBImageFromLabels(labels,centers):
    # create empty image which will be coloured
    labelsShape = labels.shape
    numRows = labelsShape[0]
    numColumns = labelsShape[1]
    rgbImage = np.zeros((numRows,numColumns,3), dtype='uint8')

    # colour the image
    rgbImage = centers[labels]
    rgbImage = rgbImage.reshape(rgbImage.shape)
    return rgbImage

################################################################################

def createSegmentedImageScatterPlot(inputImage,labelsImage,k,centers):
    # configure 3d plot format
    figure = plt.figure()
    axes = figure.add_subplot(111, projection='3d')

    # reshape the input image and labels so that RGB channels are arranged in columns
    inputImageReshaped = np.copy(inputImage)
    inputImageReshaped = inputImageReshaped.reshape((-1,3))

    labelsImageReshaped = np.copy(labelsImage)
    labelsImageReshaped = labelsImageReshaped.reshape(-1)

    # since there may be millions of data points,
    # downsample the image to only plot a max number of pixels
    numPixelsToPlot = 20000
    numPixels = inputImage.size

    if numPixels < numPixelsToPlot:
        sample = 1
    else:
        sample = round(numPixels/numPixelsToPlot)


    # create a scatter plot for each label using the original (non-segmented) image pixel data
    for x in range(0,k):

        # extract the row indices belonging to pixels in each cluster
        logicalMask = (labelsImageReshaped == x)
        maskIndices = np.nonzero(logicalMask)
        maskIndicesRows = maskIndices[0];

        # compare the pixel values across the entire sample with the values in the labels array
        maskIndicesDownsampled = maskIndicesRows[::sample]
        maskFeatures = inputImageReshaped[maskIndicesDownsampled,:]

        # split into separate RGB features for plotting
        maskFeaturesR = maskFeatures[:,0]
        maskFeaturesG = maskFeatures[:,1]
        maskFeaturesB = maskFeatures[:,2]

        axes.scatter(maskFeaturesR, maskFeaturesG, maskFeaturesB, color=centers[x,:]/255)
    # end plot

    axes.set_xlabel('R Channel')
    axes.set_ylabel('G Channel')
    axes.set_zlabel('B Channel')
    plotTitle = "Scatter Plot - RGB Image Pixel Intensities with Colour Labels"
    plt.title(plotTitle)
    plt.savefig(processingFolder + os.path.sep + plotTitle + ".png")
    plt.close()

################################################################################

# save each masked section of the segmented image separately
def saveImageSegments(segmentedImageRGB, labelsImage, labelColors, k, backgroundColor, fileNames=None):
    counter = 0 # counter to count number of segments saved
    for x in range(0,k):
        mask = np.copy(segmentedImageRGB)
        mask[labelsImage != x] = backgroundColor
        mask = mask.reshape(segmentedImageRGB.shape)

        # skip over any empty masks (e.g. black segments after blacks are removed which are empty)
        if np.all(mask == backgroundColor):
            continue
        counter += 1

        if fileNames == None:
            fileName = "Segment " + str(counter) #+ " with RGB value " + str(labelColors[x])
        else:
            fileName = fileNames[x]

        saveProcessingStep(mask,processingFolder,fileName)
        saveImageTiff(mask,profile,segmentsFolder,fileName)

################################################################################

def removeBlackPixels(labelsImage,neighbourRadius,centers,k,threshold):

    # get image of labels from segmentation that we will modify
    newLabels = np.copy(labelsImage)

    # pad image with border
    top = neighbourRadius
    bottom = neighbourRadius
    left = neighbourRadius
    right = neighbourRadius

    # extract the label number of the darkest (black) pixels
    blackPixelLabel = 0 # assume black label is label 0
    R = centers[0,0]
    G = centers[0,1]
    B = centers[0,2]
    grayLow = (int(R)+int(G)+int(B))/3

    # go through other centers and get the lowest valued gray
    for x in range(1,k):
        R = centers[x,0]
        G = centers[x,1]
        B = centers[x,2]
        grayNew = (int(R)+int(G)+int(B))/3

        # set black label to this label if it is darker than previous black label
        if grayNew < grayLow:
            grayLow = grayNew
            blackPixelLabel = x

    # do nothing and exit function if the lowest gray value is larger than the threshold
    if grayLow > threshold:
        return newLabels

    # create image padded with border (set to the black pixel label value)
    borderType = cv2.BORDER_CONSTANT
    labelsPadded = cv2.copyMakeBorder(newLabels, top, bottom, left, right, borderType, None, blackPixelLabel)

    # extract the row and column indices of every black pixel in the image
    blackPixelIndices = np.nonzero(newLabels == blackPixelLabel)

    blackPixelIndicesRow = blackPixelIndices[0]
    blackPixelIndicesColumn = blackPixelIndices[1]

    numBlackPixels = blackPixelIndicesRow.size

    # replace all padded black pixel labels with not-a-number (NAN) to allow the mode function to ignore them
    blackPixelIndicesPadded = np.nonzero(labelsPadded == blackPixelLabel)
    labelsPaddedNan = np.float32(labelsPadded)
    labelsPaddedNan[blackPixelIndicesPadded] = np.nan

    # iterate through every black pixel in the image
    for p in range(0,numBlackPixels):

        # store pixel coordinates of this iteration
        i = blackPixelIndicesRow[p]
        j = blackPixelIndicesColumn[p]

        # offset the padded image indices based on the border size (same as neighbourhood radius)
        paddedImagePixelRow = i+neighbourRadius
        paddedImagePixelColumn = j+neighbourRadius

        # extract border pixel indices of the square window around the target
        leftPixelIndex = paddedImagePixelColumn - neighbourRadius
        rightPixelIndex = paddedImagePixelColumn + neighbourRadius
        topPixelIndex = paddedImagePixelRow - neighbourRadius
        bottomPixelIndex = paddedImagePixelRow + neighbourRadius

        # extract indices of all pixels inside the window
        imageWindow = labelsPaddedNan[
            topPixelIndex:bottomPixelIndex+1,
            leftPixelIndex:rightPixelIndex+1]

        # get the mode output label (highest number of pixels with that label) and set it as a new label value
        modeOutput = stats.mode(imageWindow,axis=None,nan_policy='omit')
        modeArray = modeOutput[0]
        newLabel = modeArray[0]
        newLabels[i,j] = newLabel

    return newLabels

################################################################################

def denoiseImageLabels(labels,centers):
    labelsCopy = np.copy(labels)
    shp = labels.shape

    # create a new image with three channels (required for the ModeFilter)
    numRows = shp[0]
    numColumns = shp[1]
    numChannels = 3
    labelsCopyRGB = np.zeros((numRows,numColumns,numChannels), dtype='uint8')

    labelsCopyRGB[:,:,0] = labelsCopy
    labelsCopyRGB[:,:,1] = labelsCopy
    labelsCopyRGB[:,:,2] = labelsCopy

    # create a pillow image object from the segmented image array
    imageObject = Image.fromarray(labelsCopyRGB)

    # run the mode filter on the image multiple times to denoise
    imageObjectFiltered = imageObject.filter(ImageFilter.ModeFilter)
    for i in range(0,10):
        imageObjectFiltered = imageObjectFiltered.filter(ImageFilter.ModeFilter)

    # after filtering, create the final RGB to store the new label information for all pixels
    imageDenoised = np.asarray(imageObjectFiltered)
    imageDenoised = imageDenoised[:,:,0]

    return imageDenoised

################################################################################

def createShapeFiles():
    # create shape file output folder
    if not os.path.exists(shapeFileFolder):
        os.makedirs(shapeFileFolder)

    # create shape files for every segment in the segments folder
    imageList = os.listdir(segmentsFolder)
    for x in range(0,len(imageList)):
        # get image file name and remove file extension
        tiffFileName = imageList[x]
        temp = tiffFileName.split(".")
        tiffFileNameNoExtension = temp[0]

        shapeFileName = tiffFileNameNoExtension + ".shp"

        raster_file = segmentsFolder + os.path.sep + tiffFileName
        vector_file = shapeFileFolder + os.path.sep + shapeFileName
        driver = "ESRI Shapefile"
        mask_value = None

        rasterio_polygonize.main(raster_file, vector_file, driver, mask_value)

################################################################################
def printToLog(text):
    filePath = imageFolder + os.path.sep + 'log.txt'
    with open(filePath,"a+") as f:
        f.write(text+"\n")

################################################################################

def printProcess(fileName):
    print("Processing: " + fileName)

################################################################################

def sharpenImage(imageArray):
    # from https://stackoverflow.com/questions/32454613/python-unsharp-mask
    imageBlur = cv2.GaussianBlur(imageCropRGBCopy, (9,9), 0)
    imageUnsharp = cv2.addWeighted(imageArray, 1.5, imageBlur, -0.5, 0, imageArray)

    return imageUnsharp

################################################################################

def getBlackThresholdFromUser(image,plotTitle,initialThreshold,nbins):
    threshold = initialThreshold # start with default value
    userInputThresh = "n"
    while userInputThresh != "y": # until user says yes to continue

        # create, show and save histogram
        plt.clf() # clear figure for new plot
        plt.hist(image.flatten(),bins=nbins)
        plt.axvline(x=threshold, ymin=0, ymax=1,c='black')

        # plot intensities of the legend colors for visual comparison
        #plt.axvline(x=matureTimberGray,         ymin=0, ymax=1,c=matureTimberRGB/255)
        #plt.axvline(x=immatureTimberGray,       ymin=0, ymax=1,c=immatureTimberRGB/255)
        #plt.axvline(x=notRestockedGray,         ymin=0, ymax=1,c=notRestockedRGB/255)
        #plt.axvline(x=nonCommercialStandsGray,  ymin=0, ymax=1,c=nonCommercialStandsRGB/255)
        #plt.axvline(x=nonForestedLandGray,      ymin=0, ymax=1,c=nonForestedLandRGB/255)
        #plt.axvline(x=waterGray,                ymin=0, ymax=1,c=waterRGB/255)
        #plt.axvline(x=boundaryGray,             ymin=0, ymax=1,c=boundaryRGB/255)

        plt.title(plotTitle)
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Number of Pixels")
        plt.xlim(0,nbins-1)
        plt.show(block=False)
        plt.savefig(processingFolder + os.path.sep + plotTitle + ".png")

        print("\nA threshold of ", threshold, " has been applied to segment black/dark pixels.")
        userInputThresh = input("Enter 'y' to continue with this value or enter a new threshold between 0 and 255.\n")

        # TODO add input validation to check that input is a char or integer with proper values
        if userInputThresh.isnumeric():
            threshold = int(userInputThresh)

    return threshold

################################################################################

#-----# use k-means based image segmentation process    #-----#
def kmeansSegmentationProcess(inputImage):

    #-----#    sharpen image    #-----#
    fileName = "Sharpened RGB Image"
    printProcess(fileName)
    imageSharp = sharpenImage(inputImage)
    saveProcessingStep(imageSharp,processingFolder,fileName)

    #-----#    convert image to grayscale    #-----#
    fileName = "Grayscale of RGB Image"
    printProcess(fileName)
    imageGray = cv2.cvtColor(imageSharp, cv2.COLOR_RGB2GRAY)
    saveProcessingStep(imageGray,processingFolder,fileName)

    # show histogram of grayscale image with and let user choose threshold level
    plotTitle = "Pixel Intensity Histogram - Grayscale of Input Image"
    printProcess(plotTitle)
    nbins = 256
    initialThreshold = 80
    threshold = getBlackThresholdFromUser(imageGray,plotTitle,initialThreshold,nbins)

    printToLog("threshold: " + str(threshold))

    # let user select 'k' value
    print("\nA 'k' value is needed to segment the image into 'k' clusters. The number of peaks shown in the histogram can be used as an estimate of this value.")
    userInputK = input("Enter the number of clusters to use:\n")

    # TODO add input validation to check that input is a positive integer
    if userInputK.isnumeric():
        k = int(userInputK)
    print("")
    # close histogram plot
    plt.close()
    printToLog("k-value: " + str(k))

    #-----#    save the binary image    #-----#
    # classify pixels as either above threshold (white, value: 1) or below (black, value: 0)
    imageBW = imageGray > threshold
    fileName = "Binarized Image"
    printProcess(fileName)
    saveProcessingStep(imageBW,processingFolder,fileName)

    #-----#    set dark pixels to true black    #-----#
    fileName = "Modified RGB Image with True Black Pixels"
    printProcess(fileName)

    # extract the channels from the cropped image
    imageCropR = imageCropRGBCopy[:,:,0]
    imageCropG = imageCropRGBCopy[:,:,1]
    imageCropB = imageCropRGBCopy[:,:,2]

    # use binary image to set dark pixels to be true black
    imageCropR[imageBW==0]=0
    imageCropG[imageBW==0]=0
    imageCropB[imageBW==0]=0

    # recombine modified channels into a single RGB image
    imageCropMod = imageCropRGBCopy
    imageCropMod[:,:,0]=imageCropR
    imageCropMod[:,:,1]=imageCropG
    imageCropMod[:,:,2]=imageCropB

    saveProcessingStep(imageCropMod,processingFolder,fileName)

    #-----#    segment image    #-----#
    fileName = "Segmented Image Labels"
    printProcess(fileName)
    labelsImage, centers = segmentImage(imageCropMod,k)
    scaledLabelsImage = np.uint8(labelsImage*255/k)
    saveProcessingStep(scaledLabelsImage,processingFolder,fileName)

    #-----#    create segmented image in colour   #-----#
    fileName = "Segmented Image"
    printProcess(fileName)
    imageSegmented = createRGBImageFromLabels(labelsImage,centers)
    saveProcessingStep(imageSegmented,processingFolder,fileName)

    #-----#    show intensity distribution of segmented image    #-----#
    fileName = "Grayscale of Segmented Image"
    printProcess(fileName)
    # convert image to grayscale
    imageSegmentedGray = cv2.cvtColor(imageSegmented, cv2.COLOR_RGB2GRAY)
    saveProcessingStep(imageSegmentedGray,processingFolder,fileName)

    # configure histogram plot and show the distribution of pixels
    plotTitle = "Pixel Intensity Histogram - Grayscale of Segmented Image"
    printProcess(plotTitle)
    plt.hist(imageSegmentedGray.flatten(),bins=256)
    plt.title(plotTitle)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Number of Pixels")
    plt.xlim(0,255)
    plt.savefig(processingFolder + os.path.sep + plotTitle + ".png")
    plt.close()

    plotTitle = "Scatter Plot - RGB Image Pixel Intensities with Colour Labels"
    printProcess(plotTitle)
    createSegmentedImageScatterPlot(imageCropRGB,labelsImage,k,centers)

    #-----#    remove black pixels from the segmented image    #-----#
    fileName = "RGB Image with Black Pixels Removed"
    printProcess(fileName)
    # define a pixel neighbourhood radius
    neighbourRadius = 30
    newLabels = removeBlackPixels(labelsImage,neighbourRadius,centers,k,threshold)
    imageRGBNoBlack = createRGBImageFromLabels(newLabels,centers)
    saveProcessingStep(imageRGBNoBlack,processingFolder,fileName)

    #-----#    denoise image labels    #-----#
    fileName = "RGB Image with ModeFilter applied to reduce pixel noise"
    printProcess(fileName)
    denoisedLabels = denoiseImageLabels(newLabels,centers)
    imageRGBNoBlackDenoised = createRGBImageFromLabels(denoisedLabels,centers)
    saveProcessingStep(imageRGBNoBlackDenoised,processingFolder,fileName)

    #-----#    save image segments as separate tiff files with profile data    #-----#
    fileName = "Image Segments"
    printProcess(fileName)
    backgroundColor = [0,0,0] # white background
    saveImageSegments(imageRGBNoBlackDenoised,denoisedLabels,centers,k,backgroundColor,'none')

    #-----#    create shape files    #-----#
    fileName = "Shape files"
    printProcess(fileName)
    createShapeFiles()

################################################################################

def createHsvImageScatterPlot(imageRGB):

    # get pixel colors for plotting
    pixelColours = imageRGB.reshape((np.shape(imageRGB)[0]*np.shape(imageRGB)[1], 3))
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixelColours)
    pixelColours = norm(pixelColours).tolist()

    imageHSV = cv2.cvtColor(imageRGB, cv2.COLOR_RGB2HSV)
    imageH = imageHSV[:,:,0]
    imageS = imageHSV[:,:,1]
    imageV = imageHSV[:,:,2]

    # downsample image to fit a reasonable number of points on the plot
    numPixelsToPlot = 30000
    numPixels = imageRGB[:,:,0].size

    if numPixels < numPixelsToPlot:
        sample = 1
    else:
        sample = round(numPixels/numPixelsToPlot)

    imageH = imageH.flatten()
    imageH = imageH[::sample]

    imageS = imageS.flatten()
    imageS = imageS[::sample]

    imageV = imageV.flatten()
    imageV = imageV[::sample]

    pixelColours = pixelColours[::sample]

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    axis.scatter(imageH, imageS, imageV, facecolors=pixelColours, marker=".")
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    plotTitle = "Scatter Plot - HSV Pixel Intensities with RGB Colour Labels"
    plt.savefig(processingFolder + os.path.sep + plotTitle + ".png")
    plt.close()

################################################################################

def multiThresholdHsvSegmentationProcess(inputImage):
    # see https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_multiotsu.html
    # for example on multithresholding

    # https://docs.opencv.org/3.4/df/d9d/tutorial_py_colorspaces.html
    # see https://realpython.com/python-opencv-color-spaces/ for HSV segmentation
    # which explains how HSV can be used

    #-----#    sharpen image    #-----#
    fileName = "Sharpened RGB Image"
    printProcess(fileName)
    imageSharp = sharpenImage(inputImage)
    saveProcessingStep(imageSharp,processingFolder,fileName)

    #-----#    convert RGB to HSV    #-----#
    fileName = "HSV Image"
    printProcess(fileName)
    imageHSV = cv2.cvtColor(inputImage, cv2.COLOR_RGB2HSV)
    saveProcessingStep(imageHSV,processingFolder,fileName)

    # extract the separate HSV channels
    imageH = imageHSV[:,:,0]
    imageS = imageHSV[:,:,1]
    imageV = imageHSV[:,:,2]

    stepName = "Scatter Plot - HSV Pixel Intensities with RGB Colour Labels"
    printProcess(stepName)
    createHsvImageScatterPlot(inputImage)

    #-----#    show Value image    #-----#
    fileName = "Value Image"
    printProcess(fileName)
    saveProcessingStep(imageV,processingFolder,fileName)

    # show histogram of Value image and let user choose threshold level
    plotTitle = "Pixel Intensity Histogram - Image Value"
    printProcess(plotTitle)
    nbins = 256
    #initialThreshold = 95
    #threshold = getBlackThresholdFromUser(imageV,plotTitle,initialThreshold,nbins)

    # create, show and save histogram
    threshold = 95
    plt.hist(imageV.flatten(),bins=nbins)
    plt.axvline(x=threshold, ymin=0, ymax=1,c='black')
    plt.title(plotTitle)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Number of Pixels")
    plt.xlim(0,nbins-1)
    plt.show(block=False)
    plt.savefig(processingFolder + os.path.sep + plotTitle + ".png")

    printToLog("threshold: " + str(threshold))

    #-----#    save the binary image    #-----#
    # classify pixels as either above threshold (white, value: 1) or below (black, value: 0)
    fileName = "Binarized Image"
    printProcess(fileName)
    imageBW = imageV >= threshold
    saveProcessingStep(imageBW,processingFolder,fileName)

    def segmentByHsvRange(imageHSV,lowerBound,upperBound,segmentRGB,imageName):
        # see https://stackoverflow.com/questions/48109650/how-to-detect-two-different-colors-using-cv2-inrange-in-python-opencv

        # create mask that finds all pixels within the specified HSV range
        mask = cv2.inRange(imageHSV,lowerBound,upperBound)

        maskShape = mask.shape
        maskHeight = maskShape[0]
        maskWidth = maskShape[1]
        numChannels = 3

        # create an HSV image that is entirely one colour (the segment colour)
        segmentColourImage = np.ones((maskHeight,maskWidth,numChannels), dtype='uint8')
        segmentColourImage[:,:,0] *= segmentRGB[0]
        segmentColourImage[:,:,1] *= segmentRGB[1]
        segmentColourImage[:,:,2] *= segmentRGB[2]

        # use mask to create segmented colour image
        imageSegment = cv2.bitwise_and(segmentColourImage,segmentColourImage,mask=mask)
        #saveProcessingStep(imageSegment,processingFolder,imageName)

        # create logical mask of ones and zeros instead of 255 and 0
        mask = np.uint8(mask == 255)
        return imageSegment,mask

    fileName = "Segmented Image"
    printProcess(fileName)

    matureTimberLBound = (35, 91, threshold)
    matureTimberUBound = (60, 255, 255)
    seg1,mask1 = segmentByHsvRange(imageHSV,
        matureTimberLBound,
        matureTimberUBound,
        matureTimberRGB,
        matureTimberStr)

    immatureTimberLBound = (22, 101, threshold)
    immatureTimberUBound = (34, 255, 220)
    seg2,mask2 = segmentByHsvRange(imageHSV,
        immatureTimberLBound,
        immatureTimberUBound,
        immatureTimberRGB,
        immatureTimberStr)

    notRestockedLBound = (22, 101, 225)
    notRestockedUBound = (34, 255, 255)
    seg3,mask3 = segmentByHsvRange(imageHSV,
        notRestockedLBound,
        notRestockedUBound,
        notRestockedRGB,
        notRestockedStr)

    nonCommercialStandsLBound = (80, 101, threshold)
    nonCommercialStandsUBound = (100, 255, 255)
    seg4,mask4 = segmentByHsvRange(imageHSV,
        nonCommercialStandsLBound,
        nonCommercialStandsUBound,
        nonCommercialStandsRGB,
        nonCommercialStandsStr)

    nonForestedLandLBound = (10, 0, threshold)
    nonForestedLandUBound = (23, 100, 255)
    seg5,mask5 = segmentByHsvRange(imageHSV,
        nonForestedLandLBound,
        nonForestedLandUBound,
        nonForestedLandRGB,
        nonForestedLandStr)

    waterLBound = (28, 0, threshold)
    waterUBound = (79, 90, 255)
    seg6,mask6 = segmentByHsvRange(imageHSV,
        waterLBound,
        waterUBound,
        waterRGB,
        waterStr)

    imageSegmented = seg1 + seg2 + seg3 + seg4 + seg5 + seg6
    saveProcessingStep(imageSegmented,processingFolder,fileName)

    #-----#    create labels array from masks    #-----#
    fileName = "Segmented Image Labels"
    numSegments = 7 # including black segment
    imageLabels = mask1 + mask2*2 + mask3*3 + mask4*4 + mask5*5 + mask6*6
    imageLabelsScaled = np.uint8(imageLabels/np.max(imageLabels)*255)
    saveProcessingStep(imageLabelsScaled,processingFolder,fileName)

    # define colors (centers) of all segments as a single array
    blackRGB = np.array([0,0,0], dtype='uint8')
    centers = np.vstack((blackRGB,
        matureTimberRGB,
        immatureTimberRGB,
        notRestockedRGB,
        nonCommercialStandsRGB,
        nonForestedLandRGB,
        waterRGB))

    # define class names for all segments as a single array
    classNames = [boundaryStr,
        matureTimberStr,
        immatureTimberStr,
        notRestockedStr,
        nonCommercialStandsStr,
        nonForestedLandStr,
        waterStr]

    #-----#    remove black pixels from the segmented image    #-----#
    fileName = "RGB Image with Black Pixels Removed"
    printProcess(fileName)
    # define a pixel neighbourhood radius
    neighbourRadius = 30
    newLabels = removeBlackPixels(imageLabels,neighbourRadius,centers,numSegments,threshold)
    imageRGBNoBlack = createRGBImageFromLabels(newLabels,centers)
    saveProcessingStep(imageRGBNoBlack,processingFolder,fileName)

    #-----#    denoise image labels    #-----#
    fileName = "RGB Image with ModeFilter applied to reduce pixel noise"
    printProcess(fileName)
    denoisedLabels = denoiseImageLabels(newLabels,centers)
    imageRGBNoBlackDenoised = createRGBImageFromLabels(denoisedLabels,centers)
    saveProcessingStep(imageRGBNoBlackDenoised,processingFolder,fileName)

    #-----#    save image segments as separate tiff files with profile data    #-----#
    fileName = "Image Segments"
    printProcess(fileName)
    backgroundColor = [0,0,0] # white background
    saveImageSegments(imageRGBNoBlackDenoised,denoisedLabels,centers,numSegments,backgroundColor,classNames)

    #-----#    create shape files    #-----#
    fileName = "Shape files"
    printProcess(fileName)
    createShapeFiles()


    # # extract indices of color (non-black) pixels
    # colorPixelIndices = np.nonzero(imageBW)
    # colorPixelRows = colorPixelIndices[0]
    # colorPixelColumns = colorPixelIndices[1]
    # numColorPixels = colorPixelRows.size
    #
    # #-----#    extract the Hue of the image    #-----#
    # fileName = "Hue of Input Image"
    # printProcess(fileName)
    # imageHSV = cv2.cvtColor(inputImage, cv2.COLOR_RGB2HSV)
    # imageH = imageHSV[:,:,0]
    # saveProcessingStep(np.uint8(imageH/179*255),processingFolder,fileName)
    #
    # # create histogram of hue values for color pixels only
    # plotTitle = "Pixel Hue Histogram"
    # printProcess(plotTitle)
    # imageHColorsOnly = imageH[colorPixelRows,colorPixelColumns]
    # print(imageHColorsOnly.shape)
    #
    # # let user choose number of sections to split the histogram into
    # numClasses = 3 # start with default value
    # userInputClases = "n"
    # while userInputClases != "y":
    #
    #     # plot hsv histogram with color map
    #     # see https://stackoverflow.com/questions/23061657/plot-histogram-with-colors-taken-from-colormap
    #
    #     # get hsv color map
    #     cm = plt.cm.get_cmap('hsv')
    #
    #     # H has range of 0-179, so use 180 bins
    #     nbins = 180
    #
    #     # create histogram
    #     plt.clf() # clear figure for new plot
    #     n, bins, patches = plt.hist(imageHColorsOnly,bins=nbins)
    #     bin_centers = 0.5 * (bins[:-1] + bins[1:])
    #
    #     # scale values to interval [0,1]
    #     col = bin_centers - min(bin_centers)
    #     col /= max(col)
    #
    #     for c, p in zip(col, patches):
    #         plt.setp(p, 'facecolor', cm(c))
    #
    #     # show plot and continue executing code
    #     plt.show(block=False)
    #
    #     # apply multi-Otsu threshold
    #     thresholds = threshold_multiotsu(imageHColorsOnly,classes=numClasses,nbins=nbins)
    #
    #     # show thresholds on plot
    #     for x in range(0,len(thresholds)):
    #         plt.axvline(x=thresholds[x], ymin=0, ymax=1,c='black')
    #
    #     print(thresholds)
    #     plt.title(plotTitle)
    #     plt.xlabel("Pixel Intensity")
    #     plt.ylabel("Number of Pixels")
    #     plt.xlim(0,nbins)
    #     plt.show(block=False)
    #     plt.savefig(processingFolder + os.path.sep + plotTitle + ".png")
    #
    #     print("\nThe hue image has been sectioned into", numClasses, "classes.")
    #     userInputClases = input("Enter 'y' to continue with this value or enter a new number of classes.\n")
    #
    #     # TODO add input validation to check that input is a char or integer with proper values
    #     if userInputClases.isnumeric():
    #         numClasses = int(userInputClases)
    #
    #
    # printToLog("number of classes: " + str(numClasses))
    #
    # # extract pixel indices for each section
    # sectionIndices = []
    # for x in range(0,numClasses):
    #     sectionIndices.append(np.nonzero(imageH < thresholds[x]))

################################################################################

print("\nTEAM FOURTRESS ICS CAPSTONE PROJECT 2020")
print("DRHMAI Chloropleth raster map segmentation and polygonization\n")


# get the directory of this python script file on the local machine
# from https://stackoverflow.com/questions/5137497/find-current-directory-and-files-directory
dirPath = os.path.dirname(os.path.realpath(__file__))

# go up one folder from source file using string split and set as root folder
root,tail = os.path.split(dirPath)

outputFolder = root + os.path.sep + "output"

# set source file location for the python module
modulePath = root + os.path.sep + "source"

# add the polygonize module path to the python system path so that it can be accessed for import
# see https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
sys.path.insert(1, modulePath)
import rasterio_polygonize

# get list of all input images to process
imageDirectory = root + os.path.sep + "input"
print("Looking for images in:\n    ",imageDirectory)
imageList = sorted(os.listdir(imageDirectory))

# TODO input validation in case there are no images or if the files are not geotiffs
print("The following images were found:\n")
numImages = len(imageList)
for x in range(0,len(imageList)):
    print("    ", x+1, ":  ", imageList[x])

# get image number as input from user and convert to 0-based integer
imageNumberString = input("\nEnter the image number to process: "+ "\n")
imageNumber = int(imageNumberString) - 1

# get image name and print
imageFileNameWithExtension = imageList[imageNumber]
temp = imageFileNameWithExtension.split(".")
imageFileNameNoExtension = temp[0]
imageFilePath = imageDirectory + os.path.sep + imageFileNameWithExtension
print("\nReading File: ",imageFileNameWithExtension)

#-----#    read image    #-----#
fileName = "Input Image"
printProcess(fileName)
imageRGB,profile = readGeoTiff(imageFilePath)

# create image-specific folder directories to store processing and output images
imageFolder = outputFolder + os.path.sep + imageFileNameNoExtension
processingFolder = imageFolder + os.path.sep + "processing"
shapeFileFolder = imageFolder + os.path.sep + "shapefiles"
segmentsFolder = imageFolder + os.path.sep + "segments"

# remove existing image output folder and create a fresh clean output folder
if os.path.exists(imageFolder):
    shutil.rmtree(imageFolder)
os.makedirs(imageFolder)

printToLog("input image: " + str(imageFileNameWithExtension))
printToLog("process start: " + str(datetime.datetime.now()))

# save image and document this processing step
#saveProcessingStep(imageRGB,processingFolder,fileName)

#-----#    crop image    #-----#
imageShape = imageRGB.shape
height = imageShape[0]
width = imageShape[1]

#TODO input validation of user value
print("\nImage ",imageFileNameNoExtension, " is ", height,"by",width, " pixels in size and may require long processing times. Would you like to crop it before processing?")
userInputCrop = input("Enter 'n' to continue without cropping or enter an integer to crop the image to a square:\n")
printToLog("image height: "+ str(height))
printToLog("image width: " + str(width))

# set cropping options
if userInputCrop.isnumeric():
    value = int(userInputCrop)
    cropWidth = value
    cropHeight = value
    fileName = "Cropped Image"
else:
    cropWidth = width
    cropHeight = height
    fileName = "Input Image"

print("")
printToLog("crop height: "+ str(cropHeight))
printToLog("crop width: " + str(cropWidth))

printProcess(fileName)
# set crop size and crop the entire image
imageCropRGB = imageRGB[0:cropWidth,0:cropHeight,:]
imageCropRGBCopy = np.copy(imageCropRGB)
saveProcessingStep(imageCropRGBCopy,processingFolder,fileName)

# ask user to choose segmentation method
print("Select segmentation method:")
print("    1: gray-level thresholding with k-means segmentation")
print("    2: brightness thresholding with HSV segmentation")

# TODO: add input validation
methodNumInput = input("")


# start segmentation process
if methodNumInput.isnumeric():
    methodNum = int(methodNumInput)
    if methodNum == 1:
        kmeansSegmentationProcess(imageCropRGBCopy)

    elif methodNum == 2:# start segmentation process
        multiThresholdHsvSegmentationProcess(imageCropRGBCopy)

print("Finished!")
print("All output files have been saved to:\n    ",imageFolder)

printToLog("process finish: " + str(datetime.datetime.now()))
