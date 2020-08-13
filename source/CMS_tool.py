"""
This module contains the source code for a command-line utility that performs
image segmentation and file conversion on scanned geoTIFF choropleth maps.

Developed for the NFIS team at the Pacific Forestry Centre as part of a
Camosun College ICS Capstone Project, May-August 2020.
Author: K. Chaurasia, Team FourTrees
"""

# standard library imports
import os
import shutil
import sys
import datetime
import warnings
from pathlib import Path

# third party imports
import cv2
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageFilter
from scipy import stats
import fiona
import pandas
import geopandas

# import rasterio with warnings disabled to suppress the following warning:
# FutureWarning: GDAL-style transforms are deprecated and will not be supported in Rasterio 1.0
import rasterio
from rasterio.features import shapes
warnings.simplefilter(action='ignore', category=FutureWarning)


# define legend pixel colour values (extracted using the legend_colours.py module)
# for referencing during the segmentation process
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
# nonCommercialStandsRGB  = np.array([ 98, 153, 151], dtype='uint8')
# nonForestedLandRGB      = np.array([240, 221, 191], dtype='uint8')
# waterRGB                = np.array([203, 200, 175], dtype='uint8')
boundaryRGB             = np.array([ 77,  70,  61], dtype='uint8')

# the colours below have not been extracted from the legend but are instead
# extracted from actual map regions are more aesthetically pleasing
nonCommercialStandsRGB  = np.array([ 64, 138, 177], dtype='uint8')
nonForestedLandRGB      = np.array([239, 231, 218], dtype='uint8')
waterRGB                = np.array([194, 211, 205], dtype='uint8')

matureTimberGray        = 99
immatureTimberGray      = 141
notRestockedGray        = 154
nonCommercialStandsGray = 134
nonForestedLandGray     = 217
waterGray               = 192
boundaryGray            = 69

matureTimberStr         = 'mature timber'
immatureTimberStr       = 'immature timber'
notRestockedStr         = 'not restocked'
nonCommercialStandsStr  = 'non-commercial stands'
nonForestedLandStr      = 'non-forested land'
waterStr                = 'water'
boundaryStr             = 'boundary'

################################################################################

def readGeoTiff(imageFilePath):
    """Read a georeferenced tiff file and return its raster and profile data.

    Args:
        imageFilePath (str): Absolute or relative file path of the geotiff file
            to be read

    Returns:
        imageRGB (np uint8 array): 3-channel RGB raster image data of geotiff
        profile (rasterio profile object): Geo-referenced profile data
    """

    # read in RGB data from file
    dataset = rasterio.open(imageFilePath)
    imageR = dataset.read(1)
    imageG = dataset.read(2)
    imageB = dataset.read(3)

    # save the geodata and coordinates of the imported file
    profile = dataset.profile

    # extract the dimensions of the image to determine the number of rows and
    # in the image RGB channels.
    shape = imageR.shape

    # create variables to store row, col, and channel sizes
    numRows = shape[0]
    numColumns = shape[1]
    numChannels = 3

    # create empty black image (np.zeros makes it black)
    imageRGB = np.zeros((numRows, numColumns, numChannels), dtype='uint8')

    # replace the empty image channels with the imported RGB channels to create
    # a single RGB image
    imageRGB[:,:,0] = imageR
    imageRGB[:,:,1] = imageG
    imageRGB[:,:,2] = imageB

    return imageRGB, profile

################################################################################

def saveImagePng(imageArray, destinationFolder, fileName):
    """Save an image as PNG with a user-specified file name and to a user-
    specified directory. If the directory does not yet exist it will be created.

    Args:
        imageArray (np uint8 array): 1- or 3-channel array with image data
        destinationFolder (str): Folder in which to save the image
        fileName (str): Name of the image file (without the .png extension)
    """

    # create new output directory if it does not yet exist
    if not os.path.exists(destinationFolder):
        os.makedirs(destinationFolder)

    imageFormat = 'png'
    savePath = destinationFolder + os.path.sep + fileName + '.' + imageFormat
    imageObject = Image.fromarray(imageArray)
    imageObject.save(savePath,format=imageFormat)

################################################################################

def saveImageTiff(imageArray, profile, destinationFolder, fileName):
    """Save an image and profile data as a TIFF with a user-specified file name
    and to a user-specified directory. If the directory does not yet exist it
    will be created.

    Args:
        imageArray (np uint8 array): 3-channel array with image data
        profile (rasterio profile object): Geo-referenced profile data
        destinationFolder (str): Folder in which to save the image
        fileName (str): Name of the image file (without the .png extension)
    """

    # create new output directory if it does not yet exist
    if not os.path.exists(destinationFolder):
        os.makedirs(destinationFolder)

    # determine shape of output image
    shape = imageArray.shape
    height = int(shape[0])
    width = int(shape[1])

    # rearrange the array from height-width-channel to channel-height-width
    # which is the format expected by dst.write()
    tiffArray = np.moveaxis(imageArray,-1,0)

    # open a gdal enviroment to use with rasterio
    with rasterio.Env():
        # modify the width and height to the newly cropped image so no errors
        # will occur
        profile.update(
            width=width,
            height=height,
            dtype=rasterio.ubyte,
            count=3,
            compress='lzw')

        # create a new file with the specific name
        savePath = destinationFolder + os.path.sep + fileName + '.tiff'
        with rasterio.open(savePath,'w',**profile) as dst:
            dst.write(tiffArray)

################################################################################

def createSegmentedImageScatterPlot(inputImage, labelsImage, colours,
                                    processingFolder):
    """Create a 3D scatter plot of an image in the RGB colour space. Each
    pixel in the plot is coloured based on its segmentation label and
    segmentation colour. The plot is saved to disk at a user-defined location
    with the following file name:
        'Scatter Plot - RGB Image Pixel Intensities with Colour Labels.png'
    Note that if the image is very large, only a small sub-sample of pixels are
    plotted.

    Args:
        inputImage (np uint8 array): 3-channel array with RGB image data
        labelsImage (np uint8 array): 1-channel array with segmentation labels
        colours (np uint8 array): Array of RGB triplets that represent the pixel
            colours of each label in labelsImage
        processingFolder (str): Folder in which to save the plot
    """

    # configure 3D plot format
    figure = plt.figure()
    axes = figure.add_subplot(111, projection='3d')

    # reshape the input image and labels so that RGB channels are arranged in
    # columns
    inputImageReshaped = np.copy(inputImage)
    inputImageReshaped = inputImageReshaped.reshape((-1,3))

    labelsImageReshaped = np.copy(labelsImage)
    labelsImageReshaped = labelsImageReshaped.reshape(-1)

    # downsample the image to plot a max number of pixels instead of millions of
    # data points
    numPixelsToPlot = 20000
    numPixels = inputImage.size

    if numPixels < numPixelsToPlot:
        sample = 1
    else:
        sample = round(numPixels/numPixelsToPlot)

    # create a scatter plot for each label using the original (non-segmented)
    # image pixel data
    k = len(colours)
    for x in range(0,k):
        # extract the row indices belonging to pixels in each cluster
        logicalMask = (labelsImageReshaped == x)
        maskIndices = np.nonzero(logicalMask)
        maskIndicesRows = maskIndices[0];

        # compare the pixel values across the entire sample with the values in
        # the labels array
        maskIndicesDownsampled = maskIndicesRows[::sample]
        maskFeatures = inputImageReshaped[maskIndicesDownsampled,:]

        # split into separate RGB features for plotting
        maskFeaturesR = maskFeatures[:,0]
        maskFeaturesG = maskFeatures[:,1]
        maskFeaturesB = maskFeatures[:,2]

        axes.scatter(maskFeaturesR, maskFeaturesG, maskFeaturesB,
                     color=colours[x,:]/255)
    # end plot

    axes.set_xlabel('R Channel')
    axes.set_ylabel('G Channel')
    axes.set_zlabel('B Channel')
    plotTitle = 'Scatter Plot - RGB Image Pixel Intensities with Colour Labels'
    plt.title(plotTitle)
    plt.savefig(processingFolder + os.path.sep + plotTitle + '.png')
    plt.close()

################################################################################

def createHsvImageScatterPlot(imageRGB, processingFolder):
    """ Create a 3D scatter plot of an image in the HSV colour space. The plot
    is saved to disk at a user-defined location with the following file name:
        'Scatter Plot - HSV Pixel Intensities with RGB Colour Labels.png'
    Note that if the image is very large, only a small sub-sample of pixels are
    plotted.

    Args:
        imageRGB (np uint8 array): 3-channel array with RGB image data
        processingFolder (str): Folder in which to save the plot
    """

    # get pixel colours for plotting
    pixelColours = imageRGB.reshape(
        (np.shape(imageRGB)[0]*np.shape(imageRGB)[1], 3))
    norm = colors.Normalize(vmin=-1., vmax=1.)
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
    axis = fig.add_subplot(1, 1, 1, projection='3d')
    axis.scatter(imageH, imageS, imageV, facecolors=pixelColours, marker='.')
    axis.set_xlabel('Hue')
    axis.set_ylabel('Saturation')
    axis.set_zlabel('Value')
    plotTitle = 'Scatter Plot - HSV Pixel Intensities with RGB Colour Labels'
    plt.savefig(processingFolder + os.path.sep + plotTitle + '.png')
    plt.close()

################################################################################

def sharpenImage(imageArray):
    """Sharpen an image to enhance edges.

    Args:
        imageArray (np uint8 array): 1 or 3-channel array with image data

    Returns:
        imageArray (np uint8 array): 1 or 3-channel array with sharpened image
            data
    """

    imageBlur = cv2.GaussianBlur(imageArray, (9,9), 0)
    imageUnsharp = cv2.addWeighted(imageArray, 1.5, imageBlur, -0.5, 0,
                                   imageArray)
    return imageUnsharp

################################################################################

def getBlackThresholdFromUser(image, plotTitle, initialThreshold, nbins,
                              processingFolder):
    """Obtain an image thresholding value from the user based on a histogram of
    an image's pixel intensities.

    The function first plots the histogram with an initial threshold value as a
    vertical bar and then repeatedly asks the user to either accept the
    previously entered value or to enter a new value. The plot is saved to disk
    as a PNG at a user-defined location with a user-defined title and file name.

    Args:
        image (np uint8 array): 1-channel array with image data
        plotTitle (str): Title of the plot which is also used as the file name
        initialThreshold (str): Initial threshold value that is shown on the
            histogram
        nbins (int): Number of bins to use with the histogram
        processingFolder (str): Folder in which to save the plot

    Returns:
        threshold (int): Final threshold value selected by the user
    """

    threshold = initialThreshold # start with default value
    userInputThresh = 'n'
    isValidNumber = False
    while (userInputThresh is not 'y'): # until user says yes to continue

        # create, show and save histogram
        plt.clf() # clear figure for new plot
        plt.hist(image.flatten(),bins=nbins)
        plt.axvline(x=threshold, ymin=0, ymax=1,c='black')

        plt.title(plotTitle)
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Number of Pixels')
        plt.xlim(0,nbins-1)
        plt.show(block=False)
        plt.savefig(processingFolder + os.path.sep + plotTitle + '.png')

        print('\nA threshold of ', threshold, ' has been applied to segment '
              'black/dark pixels.')
        userInputThresh = input("Enter 'y' to continue with this value or "
            + 'enter a new threshold between 0 and ' + str(nbins-1) + '.\n')

        # ensure that the input is valid
        if (userInputThresh.isnumeric() == True
               and int(userInputThresh) < nbins
               and int(userInputThresh) >= 0):
            threshold = int(userInputThresh)

        elif userInputThresh is not 'y':
            print('That is not a valid entry. Try again.\n')

    return threshold

################################################################################

def removeBlackPixels(labelsImage, neighbourRadius, colours, threshold):
    """Remove dark/black pixels from a segmentation labels image.

    The function removes all pixels belonging to the label associated with the
    darkest segment colour. If the darkest label colour intensity is larger than
    the user-specified threshold value then no pixels are removed and the labels
    image is returned without any modifications.

    For each dark/black pixel, a neighbourhood of surrounding pixels with size
    (2*radius + 1) x (2*radius + 1) is selected and then the mode() function is
    used on these pixels to determine a replacement label. In order to 'force'
    the mode() filter to ignore the dark pixels as part of its calculation,
    dark pixel labels are first converted to not-a-number (nan).

    Args:
        labelsImage (np uint8 array): 1-channel array with segmentation labels
        neighbourRadius (str): Radius of the neighbourhood around each dark to
            consider when applying the mode() filter
        colours (np uint8 array): Array of RGB triplets that represent the pixel
            colours of each label in labelsImage
        threshold (int): Threshold value used to separate dark pixels from
            bright pixels
    """

    # get image of labels from segmentation that we will modify
    newLabels = np.copy(labelsImage)

    # pad image with border
    top = neighbourRadius
    bottom = neighbourRadius
    left = neighbourRadius
    right = neighbourRadius

    # find the label number of the darkest (black) pixels
    blackPixelLabel = 0 # assume that label 0 is the darkest
    R = colours[0,0]
    G = colours[0,1]
    B = colours[0,2]

    # get the lowest gray level intensity based on assumption that label 0 is
    # darkest
    grayLowest = (int(R)+int(G)+int(B))/3

    # go through other colours and get the lowest value for gray in the same way
    k = len(colours)
    for x in range(1,k):
        R = colours[x,0]
        G = colours[x,1]
        B = colours[x,2]
        grayNew = (int(R)+int(G)+int(B))/3

        # compare gray values and set black label to the lower of the two
        if grayNew < grayLowest:
            grayLowest = grayNew
            blackPixelLabel = x

    # do nothing and exit function if the lowest gray value is larger than the
    # threshold (no black pixels exist)
    if grayLowest > threshold:
        return newLabels

    # create image padded with border (set to the black pixel label value)
    borderType = cv2.BORDER_CONSTANT
    labelsPadded = cv2.copyMakeBorder(newLabels, top, bottom, left, right,
        borderType, None, blackPixelLabel)

    # extract the row and column indices of every black pixel in the image
    blackPixelIndices = np.nonzero(newLabels == blackPixelLabel)
    blackPixelIndicesRow = blackPixelIndices[0]
    blackPixelIndicesColumn = blackPixelIndices[1]
    numBlackPixels = blackPixelIndicesRow.size # total black pixels in image

    # replace all padded black pixel labels with not-a-number (NAN) to allow the
    # mode function to ignore them
    blackPixelIndicesPadded = np.nonzero(labelsPadded == blackPixelLabel)
    labelsPaddedNan = np.float32(labelsPadded)
    labelsPaddedNan[blackPixelIndicesPadded] = np.nan

    # iterate through every black pixel in the image
    for p in range(0,numBlackPixels):

        # store pixel coordinates of this iteration
        i = blackPixelIndicesRow[p]
        j = blackPixelIndicesColumn[p]

        # offset the padded image indices based on the border size (same as
        # neighbourhood radius)
        paddedImagePixelRow = i + neighbourRadius
        paddedImagePixelColumn = j + neighbourRadius

        # extract border pixel indices of the square window around the target
        leftPixelIndex = paddedImagePixelColumn - neighbourRadius
        rightPixelIndex = paddedImagePixelColumn + neighbourRadius
        topPixelIndex = paddedImagePixelRow - neighbourRadius
        bottomPixelIndex = paddedImagePixelRow + neighbourRadius

        # extract indices of all pixels inside the window
        imageWindow = labelsPaddedNan[
            topPixelIndex:bottomPixelIndex+1,
            leftPixelIndex:rightPixelIndex+1]

        # get the mode output label (highest number of pixels with that label)
        # and set it as a new label value
        modeOutput = stats.mode(imageWindow,axis=None,nan_policy='omit')
        modeArray = modeOutput[0]
        newLabel = modeArray[0]
        newLabels[i,j] = newLabel

    return newLabels

################################################################################

def denoiseImageLabels(labels):
    """Denoise a segmentation labels image by repeatedly applying a mode filter.

    Args:
        labels (np uint8 array): 1-channel image array with segmentation labels

    Returns:
        labelsDenoised (np uint8 array): 1-channel image array with denoised
            segmentation labels
    """

    labelsCopy = np.copy(labels)
    shp = labels.shape

    # create a new image with three channels (required for the ModeFilter)
    numRows = shp[0]
    numColumns = shp[1]
    numChannels = 3
    labelsCopyRGB = np.zeros((numRows, numColumns, numChannels), dtype='uint8')

    # copy labels image so all three channels contain the same data
    labelsCopyRGB[:,:,0] = labelsCopy
    labelsCopyRGB[:,:,1] = labelsCopy
    labelsCopyRGB[:,:,2] = labelsCopy

    # create a pillow image object from the labels image array
    imageObject = Image.fromarray(labelsCopyRGB)

    # run the mode filter on the image multiple times to denoise
    imageObjectFiltered = imageObject.filter(ImageFilter.ModeFilter)
    for i in range(0, 10):
        imageObjectFiltered = imageObjectFiltered.filter(ImageFilter.ModeFilter)

    # after filtering, convert image object to a single channel image array
    labelsDenoised = np.asarray(imageObjectFiltered)
    labelsDenoised = labelsDenoised[:,:,0]

    return labelsDenoised

################################################################################

def segmentImageKmeans(imageArray, k):
    """Segment a colour image into 'k' segments using the k-means clustering
    algorithm. Each channel of the image is treated as a separate feature, e.g.
    an RGB image will be segmented using the R, G and B channels as features.

    Args:
        imageArray (np uint8 array): 3-channel array with image data
        k (int): The number of clusters or segments to classify the pixels into

    Returns:
        labelsImage (np uint8 array): 1-channel image array with segmentation
            labels
        colours (np uint8 array): Array of RGB triplets that represent the pixel
            colours of each segment
    """

    imageShape = imageArray.shape
    numRows = imageShape[0]
    numColumns = imageShape[1]

    # reshape image so that it has three column 32bit float features as required
    # by the kmeans function
    imageFeatures = np.float32(imageArray.reshape((-1,3)))

    # set algorithm stopping criteria (based on OpenCV k-means function
    # documentation)
    # this controls what happens if max iterations reached or if cluster
    # assignments do not change (small epsilon)
    maxIterations = 10
    epsilon = 1
    terminationType = cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS
    criteria = (terminationType, maxIterations, epsilon)
    attempts = 10
    flags = cv2.KMEANS_RANDOM_CENTERS # specify how initial centres are taken

    # segment the image with k-means and extract data about the labels and
    # centres. In this case, cluster centres are RGB colours
    compactness, labels, colours = cv2.kmeans(imageFeatures, k, None, criteria,
                                              attempts, flags)

    # format the k-means output and re-shape to create the labels image
    colours = np.uint8(colours)
    labels = labels.flatten()
    labelsImage = np.copy(labels)
    labelsImage = labelsImage.reshape((numRows, numColumns))

    return labelsImage, colours

################################################################################

def createRGBImageFromLabels(labels, colours):
    """Reconstruct a 3-channel RGB colour image from segmentation labels and
    array of colours.

    Args:
        labels (np uint8 array): 1-channel image array with segmentation labels
        colours (np uint8 array): Array of RGB triplets that represent the pixel
            colours of each segment

    Returns:
        rgbImage (np uint8 array): 3-channel RGB image array with coloured
            segments
    """

    # create empty image which will be coloured
    labelsShape = labels.shape
    numRows = labelsShape[0]
    numColumns = labelsShape[1]
    rgbImage = np.zeros((numRows, numColumns, 3), dtype='uint8')

    # colour the image
    rgbImage = colours[labels]
    rgbImage = rgbImage.reshape(rgbImage.shape)
    return rgbImage

################################################################################

# save each masked section of the segmented image separately
def saveImageSegments(segmentedImageRGB, labelsImage, profile,
                      processingFolder, segmentsFolder, fileNames=None):
    """Save each coloured segment of an RGB image as both a PNG and TIFF to
    separate user-defined directories. Segments are extracted using the
    segmentation labels; TIFFs are saved with profile data. Image file names
    can optionaly be provided, or will otherwise be saved as 'Segment #' with
    '#' increasing sequentially from '1'.

    Args:
        segmentedImageRGB (np uint8 array): 3-channel array of segmented image
        labelsImage (np uint8 array): 1-channel array of segmentation labels
        profile (rasterio profile object): Geo-referenced profile data
        processingFolder (str): Folder in which to save the PNG image
        segmentsFolder (str): Folder in which to save the TIFF image
        fileNames (str): Optional list of filenames associated with each segment
    """
    backgroundColour = [0,0,0] # define black colour for segment backgrounds
    counter = 0 # counter to count number of segments saved
    k = np.max(labelsImage) + 1 # get number of segments
    for x in range(0, k):
        mask = np.copy(segmentedImageRGB)
        mask[labelsImage != x] = backgroundColour
        mask = mask.reshape(segmentedImageRGB.shape)

        # skip over empty masks (e.g. black segments after blacks are removed -
        # these are empty)
        if np.all(mask == backgroundColour):
            continue
        counter += 1

        # files are named generically unless specific names (e.g. legend
        # categories) are available or provided for use
        if fileNames == 'none':
            fileName = 'Segment ' + str(counter)
        else:
            fileName = fileNames[x]

        saveProcessingStep(mask, processingFolder, fileName)
        saveImageTiff(mask, profile, segmentsFolder, fileName)

################################################################################

def createShapeFiles(shapeFileFolder, segmentsFolder):
    """ Converts all TIFF images in a user-specified directory to shape files
    and saves them to another user-specified directory. If the destination
    folder does not exist it will be created. The shape files are saved with the
    same file names as the original TIFF images.

    Args:
        shapeFileFolder (str): Folder in which to save the shape files
        segmentsFolder (str): Folder which contains TIFF images to be converted
    """

    # create shape file output folder
    if not os.path.exists(shapeFileFolder):
        os.makedirs(shapeFileFolder)

    # create shape files for every segment in the segments folder
    imageList = os.listdir(segmentsFolder)
    for x in range(0, len(imageList)):
        # get image file name and remove file extension
        tiffFileName = imageList[x]
        temp = tiffFileName.split('.')
        tiffFileNameNoExtension = temp[0]

        shapeFileName = tiffFileNameNoExtension + '.shp'

        rasterFile = segmentsFolder + os.path.sep + tiffFileName
        vectorFile = shapeFileFolder + os.path.sep + shapeFileName
        driver = 'ESRI Shapefile'
        maskValue = 0 # remove blacks (0s) from shapefile output

        # polygonize the raster files
        polygonize(rasterFile, vectorFile, driver, maskValue)

################################################################################

def mergeShapefiles(mergedShapeFileFolder, shapeFileFolder, fileName):
    """ Merge all shapefiles (.shp) in a user-specified directory into a single
    shapefile and save it to another user-specified directory with a specific
    file name. If the destination folder does not exist it will be created.

    Args:
        mergedShapeFileFolder (str): Folder in which to save the merged
            shapefiles
        shapeFileFolder (str): Folder which contains SHP files to be merged
        fileName (str): Name of the merged shapefile (without .shp extension)
    """

    # create merged shape file output folder
    if not os.path.exists(mergedShapeFileFolder):
        os.makedirs(mergedShapeFileFolder)

    # get list of all shape files to process
    fileList = sorted(os.listdir(shapeFileFolder))
    shapeList = []
    for x in range(0, len(fileList)):
        f = fileList[x]
        # get .shp files only
        if f.endswith('.shp'):
            # read shape file
            shp = geopandas.read_file(shapeFileFolder + os.path.sep + f)
            shapeList.append(shp) # store in list

    # concatenate shapes (i.e. merge) and save to file
    mergedShapes = pandas.concat(shapeList)
    geoDataFrame = geopandas.GeoDataFrame(mergedShapes)
    geoDataFrame.to_file(mergedShapeFileFolder + os.path.sep
                         + fileName + '.shp')

################################################################################

def polygonize(rasterFile, vectorFile, driver, maskValue):
    """Convert a geo-referenced raster file of a single-colour 3-channel image
    segment to a vectorized polygon shapefile. Each polygon region of the shape
    file is also given a 'CATEGORY' attribute with a value equivalent to the
    raster file's name. This attribute is visible from within GIS software.

    This function emulates GDAL's gdal_polygonize.py and is adapted from:
    https://github.com/sgillies/rasterio/blob/master/examples/rasterio_polygonize.py

    Args:
        rasterFile (str): File path of the TIFF raster image to be read
        vectorFile (str): File path of the shapefile to be written
        driver (str): Name of the driver to use to create the shapefile
        maskValue (int): Integer raster value to omit from the shapefile
    """
    # get image file name and remove file extension
    rasterFileNameNoExtension = Path(rasterFile).stem
    vectorFileWithExtension = os.path.basename(vectorFile)

    with rasterio.Env():
        with rasterio.open(rasterFile) as src:
            imageR = src.read(1)

        if maskValue is not None:
            mask = imageR != maskValue
        else:
            mask = None

        # create geojson file structure with geometry data
        # a CATEGORY attribute is also written to each geometry regions
        # so that the map cover category can be stored with the shapefile
        results = (
            {'properties': {'CATEGORY': rasterFileNameNoExtension}, 'geometry': s}
            for i, (s, v) in enumerate(
                shapes(imageR, mask=mask, transform=src.transform)))

        # use fiona to convert the geojson to a shapefile
        try:
            with fiona.open(
                    vectorFile, 'w',
                    driver=driver,
                    crs=src.crs,
                    schema={'properties': [('CATEGORY', 'str')],
                            'geometry': 'Polygon'}) as dst:

                    dst.writerecords(results)
        except:
            print("Could not create the shape file '"
                + vectorFileWithExtension + "'.\n")
            print('The file may be open in another program. Please close it '
                  + 'and re-start the tool.')

    return dst.name

################################################################################

def saveProcessingStep(imageArray, destinationFolder, fileName):
    """Save image to file as PNG to a user-specified directory with a
    user-specified file name. A numerical prefix is added to the image file name
    (starting at '01') and this prefix is incremented by one every time the
    function is called.

    Args:
        imageArray (np uint8 array): 1- or 3-channel array with image data
        destinationFolder (str): Folder in which to save the image
        fileName: Name of the file to be saved (without prefix or extension)
    """

    # create local counter to increment each time the function is called
    if not hasattr(saveProcessingStep, 'counter'):
        saveProcessingStep.counter = 0 # initialize attribute
    saveProcessingStep.counter += 1

    # convert counter to string with preceding 0 for single digits
    numString = str(saveProcessingStep.counter)
    if saveProcessingStep.counter < 10:
        numString = str(0) + numString

    stepName = numString + ' ' + fileName
    saveImagePng(imageArray, destinationFolder, stepName)

################################################################################

def printToLog(imageFolder, text):
    """Append text to a 'log.txt' file located in imageFolder.
    """

    filePath = imageFolder + os.path.sep + 'log.txt'
    with open(filePath, 'a+') as f:
        f.write(text+'\n')

################################################################################

def getRootFolder():
    """Return the root folder of the CMS tool repository.
    """

    # get the directory of this python script file on a local machine
    dirPath = Path(__file__).resolve().parent

    # go up one folder from source file using string split and set as root
    # folder
    root, tail = os.path.split(dirPath)
    return root

################################################################################

def printProcess(text):
    """Print text to console with a 'Processing: ' prefix to document the
    current processing steps.
    """

    print('Processing: ' + text)

################################################################################

def kmeansSegmentationProcess(inputImage, profile, imageFolder,
        processingFolder, segmentsFolder, shapeFileFolder):
    """Process an image using a series of processing steps. The image is
    segmented using the k-means method and output files are saved to multiple
    folders.

    Args:
        inputImage (np uint8 array): 3-channel array with RGB image data
        profile (rasterio profile object): Geo-referenced profile data
        imageFolder (str): Image-specific folder where all outputs are saved
        processingFolder (str): Folder where images are saved during processing
        segmentsFolder (str): Folder where final image segments are saved
        shapeFileFolder (str): Folder where shapefile outputs are saved
    """

    #-----#    sharpen image    #-----#
    fileName = 'Sharpened RGB Image'
    printProcess(fileName)
    imageSharp = sharpenImage(inputImage)
    saveProcessingStep(imageSharp, processingFolder, fileName)

    #-----#    convert image to grayscale    #-----#
    fileName = 'Grayscale of RGB Image'
    printProcess(fileName)
    imageGray = cv2.cvtColor(imageSharp, cv2.COLOR_RGB2GRAY)
    saveProcessingStep(imageGray, processingFolder, fileName)

    # show histogram of grayscale image with and let user choose threshold level
    plotTitle = 'Pixel Intensity Histogram - Grayscale of Input Image'
    printProcess(plotTitle)
    nbins = 256
    initialThreshold = 80
    threshold = getBlackThresholdFromUser(imageGray, plotTitle,
        initialThreshold, nbins, processingFolder)

    printToLog(imageFolder, 'threshold: ' + str(threshold))

    # let user select 'k' value
    print("\nA 'k' value is needed to segment the image into 'k' clusters. "
          + 'The number of peaks shown in the histogram can be used as an '
          + 'estimate of this value.')

     # ensure that the input is valid
    isValidNumber = False
    while not isValidNumber:
        userInputK = input('Enter the number of clusters to use:\n')
        if (userInputK.isnumeric() == True and int(userInputK) > 0):
            isValidNumber = True
            k = int(userInputK)
        else:
            print("That is not a valid entry. 'k' must be a positive integer. "
                  + 'Try again.\n')

    print('')
    # close histogram plot
    plt.close()
    printToLog(imageFolder, 'k-value: ' + str(k))

    #-----#    save the binary image    #-----#
    # classify pixels as either above threshold (white, value: 1) or
    # below (black, value: 0)
    imageBW = imageGray > threshold
    fileName = 'Binarized Image'
    printProcess(fileName)
    saveProcessingStep(imageBW, processingFolder, fileName)

    #-----#    set dark pixels to true black    #-----#
    fileName = 'Modified RGB Image with True Black Pixels'
    printProcess(fileName)

    # extract the channels from the input image
    imageCropR = inputImage[:,:,0]
    imageCropG = inputImage[:,:,1]
    imageCropB = inputImage[:,:,2]

    # use binary image to set dark pixels to be true black
    imageCropR[imageBW==0] = 0
    imageCropG[imageBW==0] = 0
    imageCropB[imageBW==0] = 0

    # recombine modified channels into a single RGB image
    imageCropMod = inputImage
    imageCropMod[:,:,0] = imageCropR
    imageCropMod[:,:,1] = imageCropG
    imageCropMod[:,:,2] = imageCropB

    saveProcessingStep(imageCropMod, processingFolder, fileName)

    #-----#    segment image    #-----#
    fileName = 'Segmented Image Labels'
    printProcess(fileName)
    labelsImage, colours = segmentImageKmeans(imageCropMod, k)
    scaledLabelsImage = np.uint8(labelsImage*255/k)
    saveProcessingStep(scaledLabelsImage, processingFolder, fileName)

    #-----#    create segmented image in colour   #-----#
    fileName = 'Segmented Image'
    printProcess(fileName)
    imageSegmented = createRGBImageFromLabels(labelsImage, colours)
    saveProcessingStep(imageSegmented, processingFolder, fileName)

    #-----#    show intensity distribution of segmented image    #-----#
    fileName = 'Grayscale of Segmented Image'
    printProcess(fileName)
    # convert image to grayscale
    imageSegmentedGray = cv2.cvtColor(imageSegmented, cv2.COLOR_RGB2GRAY)
    saveProcessingStep(imageSegmentedGray, processingFolder, fileName)

    # configure histogram plot and show the distribution of pixels
    plotTitle = 'Pixel Intensity Histogram - Grayscale of Segmented Image'
    printProcess(plotTitle)
    plt.hist(imageSegmentedGray.flatten(), bins=256)
    plt.title(plotTitle)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Number of Pixels')
    plt.xlim(0, 255)
    plt.savefig(processingFolder + os.path.sep + plotTitle + '.png')
    plt.close()

    plotTitle = 'Scatter Plot - RGB Image Pixel Intensities with Colour Labels'
    printProcess(plotTitle)
    createSegmentedImageScatterPlot(inputImage, labelsImage, colours,
                                    processingFolder)

    #-----#    remove black pixels from the segmented image    #-----#
    fileName = 'RGB Image with Black Pixels Removed'
    printProcess(fileName)

    # define a pixel neighbourhood radius
    neighbourRadius = 30
    newLabels = removeBlackPixels(labelsImage, neighbourRadius, colours,
                                  threshold)
    imageRGBNoBlack = createRGBImageFromLabels(newLabels, colours)
    saveProcessingStep(imageRGBNoBlack, processingFolder, fileName)

    #-----#    denoise image labels    #-----#
    fileName = 'RGB Image with ModeFilter applied to reduce pixel noise'
    printProcess(fileName)
    denoisedLabels = denoiseImageLabels(newLabels)
    imageRGBNoBlackDenoised = createRGBImageFromLabels(denoisedLabels, colours)
    saveProcessingStep(imageRGBNoBlackDenoised, processingFolder, fileName)

    #-----#    save image segments as separate tiff files with profile data    #-----#
    fileName = 'Image Segments'
    printProcess(fileName)
    backgroundColour = [0,0,0] # white background
    saveImageSegments(imageRGBNoBlackDenoised, denoisedLabels, profile,
                      processingFolder, segmentsFolder, 'none')

    #-----#    create shape files    #-----#
    fileName = 'Shape files'
    printProcess(fileName)
    createShapeFiles(shapeFileFolder, segmentsFolder)

    fileName = os.path.basename(imageFolder)
    mergeShapefiles(shapeFileFolder, shapeFileFolder, fileName)

################################################################################

def hsvSegmentationProcess(inputImage, profile, imageFolder, processingFolder,
                           segmentsFolder, shapeFileFolder):
    """Process an image using a series of processing steps. The image is
    segmented using the HSV method and output files are saved to multiple
    folders.

    Args:
        inputImage (np uint8 array): 3-channel array with RGB image data
        profile (rasterio profile object): Geo-referenced profile data
        imageFolder (str): Image-specific folder where all outputs are saved
        processingFolder (str): Folder where images are saved during processing
        segmentsFolder (str): Folder where final image segments are saved
        shapeFileFolder (str): Folder where shapefile outputs are saved
    """

    #-----#    sharpen image    #-----#
    fileName = 'Sharpened RGB Image'
    printProcess(fileName)
    imageSharp = sharpenImage(inputImage)
    saveProcessingStep(imageSharp, processingFolder, fileName)

    #-----#    convert RGB to HSV    #-----#
    fileName = 'HSV Image'
    printProcess(fileName)
    imageHSV = cv2.cvtColor(inputImage, cv2.COLOR_RGB2HSV)
    saveProcessingStep(imageHSV, processingFolder, fileName)

    # extract the separate HSV channels
    imageH = imageHSV[:,:,0]
    imageS = imageHSV[:,:,1]
    imageV = imageHSV[:,:,2]

    stepName = 'Scatter Plot - HSV Pixel Intensities with RGB Colour Labels'
    printProcess(stepName)
    createHsvImageScatterPlot(inputImage, processingFolder)

    #-----#    show Value image    #-----#
    fileName = 'Value Image'
    printProcess(fileName)
    saveProcessingStep(imageV, processingFolder, fileName)

    # show histogram of Value image and let user choose threshold level
    plotTitle = 'Pixel Intensity Histogram - Image Brightness'
    printProcess(plotTitle)
    nbins = 256
    #initialThreshold = 95
    #threshold = getBlackThresholdFromUser(imageV, plotTitle,
    #                                      initialThreshold, nbins)

    # create, show and save histogram
    threshold = 90
    plt.hist(imageV.flatten(), bins=nbins)
    plt.axvline(x=threshold, ymin=0, ymax=1, c='black')
    plt.title(plotTitle)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Number of Pixels')
    plt.xlim(0, nbins-1)
    plt.savefig(processingFolder + os.path.sep + plotTitle + '.png')
    plt.close()

    printToLog(imageFolder, 'threshold: ' + str(threshold))

    #-----#    save the binary image    #-----#
    # classify pixels as either above threshold (white, value: 1)
    # or below (black, value: 0)
    fileName = 'Binarized Image'
    printProcess(fileName)
    imageBW = imageV >= threshold
    saveProcessingStep(imageBW, processingFolder, fileName)

    # segment image by HSV bounds
    def segmentByHsvRange(imageHSV, lowerBound, upperBound, segmentRGB,
                          imageName):
        # create mask that finds all pixels within the specified HSV range
        mask = cv2.inRange(imageHSV, lowerBound, upperBound)

        maskShape = mask.shape
        maskHeight = maskShape[0]
        maskWidth = maskShape[1]
        numChannels = 3

        # create an HSV image that is entirely one colour (the segment colour)
        segmentColourImage = np.ones((maskHeight, maskWidth, numChannels),
                                     dtype='uint8')
        segmentColourImage[:,:,0] *= segmentRGB[0]
        segmentColourImage[:,:,1] *= segmentRGB[1]
        segmentColourImage[:,:,2] *= segmentRGB[2]

        # use mask to create segmented colour image
        imageSegment = cv2.bitwise_and(segmentColourImage, segmentColourImage,
                                       mask=mask)

        # create logical mask of ones and zeros instead of 255 and 0
        mask = np.uint8(mask == 255)
        return imageSegment, mask

    fileName = 'Segmented Image'
    printProcess(fileName)

    matureTimberLBound = (35, 91, threshold)
    matureTimberUBound = (82, 255, 255)
    seg1, mask1 = segmentByHsvRange(imageHSV,
        matureTimberLBound,
        matureTimberUBound,
        matureTimberRGB,
        matureTimberStr)
    printToLog(imageFolder, matureTimberStr
        + ' LB: {0}'.format(matureTimberLBound))
    printToLog(imageFolder, matureTimberStr
        + ' UB: {0}'.format(matureTimberUBound))

    immatureTimberLBound = (22, 101, threshold)
    immatureTimberUBound = (34, 255, 220)
    seg2, mask2 = segmentByHsvRange(imageHSV,
        immatureTimberLBound,
        immatureTimberUBound,
        immatureTimberRGB,
        immatureTimberStr)
    printToLog(imageFolder, immatureTimberStr
        + ' LB: {0}'.format(immatureTimberLBound))
    printToLog(imageFolder, immatureTimberStr
        + ' UB: {0}'.format(immatureTimberUBound))

    notRestockedLBound = (22, 101, 225)
    notRestockedUBound = (34, 255, 255)
    seg3, mask3 = segmentByHsvRange(imageHSV,
        notRestockedLBound,
        notRestockedUBound,
        notRestockedRGB,
        notRestockedStr)
    printToLog(imageFolder, notRestockedStr
        + ' LB: {0}'.format(notRestockedLBound))
    printToLog(imageFolder, notRestockedStr
        + ' UB: {0}'.format(notRestockedUBound))

    nonCommercialStandsLBound = (83, 101, threshold)
    nonCommercialStandsUBound = (120, 255, 255)
    seg4, mask4 = segmentByHsvRange(imageHSV,
        nonCommercialStandsLBound,
        nonCommercialStandsUBound,
        nonCommercialStandsRGB,
        nonCommercialStandsStr)
    printToLog(imageFolder, nonCommercialStandsStr
        + ' LB: {0}'.format(nonCommercialStandsLBound))
    printToLog(imageFolder, nonCommercialStandsStr
        + ' UB: {0}'.format(nonCommercialStandsUBound))

    nonForestedLandLBound = (10, 0, threshold)
    nonForestedLandUBound = (28, 95, 255)
    seg5, mask5 = segmentByHsvRange(imageHSV,
        nonForestedLandLBound,
        nonForestedLandUBound,
        nonForestedLandRGB,
        nonForestedLandStr)
    printToLog(imageFolder, nonForestedLandStr
        + ' LB: {0}'.format(nonForestedLandLBound))
    printToLog(imageFolder, nonForestedLandStr
        + ' UB: {0}'.format(nonForestedLandUBound))

    waterLBound = (30, 0, threshold)
    waterUBound = (90, 90, 255)
    seg6, mask6 = segmentByHsvRange(imageHSV,
        waterLBound,
        waterUBound,
        waterRGB,
        waterStr)
    printToLog(imageFolder,waterStr
        + ' LB: {0}'.format(waterLBound))
    printToLog(imageFolder,waterStr
        + ' UB: {0}'.format(waterUBound))

    imageSegmented = seg1 + seg2 + seg3 + seg4 + seg5 + seg6
    saveProcessingStep(imageSegmented, processingFolder, fileName)

    #-----#    create labels array from masks    #-----#
    fileName = 'Segmented Image Labels'
    numSegments = 7 # including black segment
    imageLabels = mask1 + mask2*2 + mask3*3 + mask4*4 + mask5*5 + mask6*6
    imageLabelsScaled = np.uint8(imageLabels/np.max(imageLabels)*255)
    saveProcessingStep(imageLabelsScaled, processingFolder, fileName)

    # define colours of all segments as a single array
    blackRGB = np.array([0,0,0], dtype='uint8')
    colours = np.vstack((blackRGB,
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
    fileName = 'RGB Image with Black Pixels Removed'
    printProcess(fileName)
    # define a pixel neighbourhood radius
    neighbourRadius = 30
    newLabels = removeBlackPixels(imageLabels, neighbourRadius, colours,
                                  threshold)
    imageRGBNoBlack = createRGBImageFromLabels(newLabels, colours)
    saveProcessingStep(imageRGBNoBlack, processingFolder, fileName)

    #-----#    denoise image labels    #-----#
    fileName = 'RGB Image with ModeFilter applied to reduce pixel noise'
    printProcess(fileName)
    denoisedLabels = denoiseImageLabels(newLabels)
    imageRGBNoBlackDenoised = createRGBImageFromLabels(denoisedLabels, colours)
    saveProcessingStep(imageRGBNoBlackDenoised, processingFolder, fileName)

    #-----#    save image segments as separate tiff files with profile data    #-----#
    fileName = 'Image Segments'
    printProcess(fileName)
    backgroundColour = [0,0,0] # white background
    saveImageSegments(imageRGBNoBlackDenoised, denoisedLabels, profile,
                      processingFolder, segmentsFolder, classNames)

    #-----#    create shape files    #-----#
    fileName = 'Shape files'
    printProcess(fileName)
    createShapeFiles(shapeFileFolder, segmentsFolder)

    fileName = os.path.basename(imageFolder)
    mergeShapefiles(shapeFileFolder, shapeFileFolder, fileName)

################################################################################

def processImage(imageDirectory, imageFileNameWithExtension,
                 includeUserInput=False, cropSize=None):
    """Process an image with or without user input to configure processing
    parameters. If no user input is provided, the full-sized image is processed
    using the HSV segmentation method. If user input is provided, the image may
    be cropped and either the k-means or HSV segmentation method may be
    selected.

    The function creates an output directory with the same name as the image
    being processed and saves all output files to this directory.

    Args:
        imageDirectory (str): Folder where the image to be processed is located
        imageFileNameWithExtension (str): Name of the file to be processed
        includeUserInput (bool): Optional parameter that if 'true' enables input
            from the user to configure processing options
        cropSize (int): Optional parameter to crop the image to a square of size
            cropSize by cropSize from the top-left corner of the image
    """

    # reset counter used for saving processing step images
    saveProcessingStep.counter = 0

    print('\nReading File:', imageFileNameWithExtension)
    imageFileNameNoExtension = os.path.splitext(imageFileNameWithExtension)[0]

    #-----#    read input image and set up directories for output storage   #-----#
    imageFilePath = imageDirectory + os.path.sep + imageFileNameWithExtension
    imageRGB, profile = readGeoTiff(imageFilePath)

    # create image-specific folder directories to store processing and output
    # images
    outputFolder = getRootFolder() + os.path.sep + 'output'
    imageFolder = outputFolder + os.path.sep + imageFileNameNoExtension
    processingFolder = imageFolder + os.path.sep + 'processing'
    shapeFileFolder = imageFolder + os.path.sep + 'shapefiles'
    segmentsFolder = imageFolder + os.path.sep + 'segments'

    # remove existing image output folder and create a fresh clean output folder
    #
    # On Windows, errors may be thrown if files inside imageFolder are open so
    # the ignore_errors flag is used to overcome this. When errors do exist, the
    # imageFolder itself may also not be removed, and so the exist_ok flag is
    # used when creating a new directory. Permissions are also set with 0o777.
    # A for loop is also used to retry making the folder because Windows will
    # sometimes fail to remove or make the directory despite the use of flags
    for retry in range(100):
        try:
            if os.path.exists(imageFolder):
                shutil.rmtree(imageFolder, ignore_errors=True)
            os.makedirs(imageFolder, 0o777, exist_ok=True)
            break
        except:
            donothing=1 # do nothing, keep retrying

    printToLog(imageFolder, 'input image: ' + str(imageFileNameWithExtension))
    now = datetime.datetime.now().isoformat(' ', 'seconds')
    printToLog(imageFolder, 'process start: ' + str(now))

    #-----#    crop image    #-----#
    imageShape = imageRGB.shape
    height = imageShape[0]
    width = imageShape[1]

    printToLog(imageFolder, 'image height: '+ str(height))
    printToLog(imageFolder, 'image width: ' + str(width))

    # assume crop at full size unless overriden by user input
    if cropSize==None:
        cropWidth = width
        cropHeight = height
        cropFileName = 'Input Image'
    else:
        maxCropSize = min((height, width))
        assert(cropSize <= maxCropSize and cropSize > 0)
        cropWidth = cropSize
        cropHeight = cropSize
        cropFileName = 'Cropped Image'

    # get cropping dimensions from user if only processing one file
    if includeUserInput == True:
        print('\nImage', imageFileNameNoExtension, 'is', height, 'by', width,
            'pixels in size and may require long processing times.',
            'Would you like to crop it before processing?')
        userInputCrop = input("Enter 'n' to continue without cropping "
            + 'OR enter an integer value to crop the image to a square:\n')

        # ensure that the input number is valid
        while True:
            maxCropSize = min((height, width))
            isWithinBounds = (userInputCrop.isnumeric() == True
                              and int(userInputCrop) <= maxCropSize
                              and int(userInputCrop) > 0)
            if userInputCrop == 'n':
                break # exit loop and use assumed full crop size
            elif isWithinBounds == True:
                # set crop size to user input value and exit loop
                value = int(userInputCrop)
                cropWidth = value
                cropHeight = value
                cropFileName = 'Cropped Image'
                break
            else:
                # input is bad, get new input
                userInputCrop = input('\nThat is not a valid crop value. '
                                      + 'Try again.\n')

        print('')

    printToLog(imageFolder, 'crop height: '+ str(cropHeight))
    printToLog(imageFolder, 'crop width: ' + str(cropWidth))

    # assume that method 2 will be used (HSV segmentation)
    methodNum = 2
    if includeUserInput == True:
        # ask user to choose segmentation method
        print('Select segmentation method:')
        print('     1 :  gray-level thresholding with k-means segmentation')
        print('     2 :  brightness thresholding with HSV segmentation')
        methodNumInput = input('')

        # ensure that the input number is valid
        while (methodNumInput.isnumeric() == False
               or (int(methodNumInput) != 1 and int(methodNumInput) != 2)):
            methodNumInput = input('\nThat is not a valid selection. '
                                   + 'Try again.\n')

        methodNum = int(methodNumInput)
        print('')

    # crop the image
    printProcess(cropFileName)
    imageCropRGB = imageRGB[0:cropWidth,0:cropHeight,:]
    imageCropRGBCopy = np.copy(imageCropRGB)
    saveProcessingStep(imageCropRGBCopy, processingFolder, cropFileName)

    #-----#    begin segmentation process based on user selection    #-----#
    if methodNum == 1:
        printToLog(imageFolder, 'segmentation method: k-means')
        kmeansSegmentationProcess(imageCropRGBCopy, profile, imageFolder,
            processingFolder, segmentsFolder, shapeFileFolder)

    elif methodNum == 2:# start segmentation process
        printToLog(imageFolder, 'segmentation method: HSV')
        hsvSegmentationProcess(imageCropRGBCopy, profile, imageFolder,
            processingFolder, segmentsFolder, shapeFileFolder)

    print('Finished segmenting ' + imageFileNameNoExtension + '!')
    print('All output files have been saved to:\n    ', imageFolder)

    now = datetime.datetime.now().isoformat(' ', 'seconds')
    printToLog(imageFolder, 'process end: ' + str(now))

################################################################################

def run():
    """Runs the CMS tool. A welcome message is printed to console and TIFF
    images are read from the 'input' directory then displayed in a list to the
    user. The user is then prompted for input on how to proceed.
    """

    print('\nDigital Resurrection of Historical Maps using '
        + 'Artificial Intelligence (DRHMAI) Project')
    print('Team FourTrees, Camosun College ICS Capstone 2020\n')
    print('Welcome to the Choropleth Map Segmentation Tool!\n')

    root = getRootFolder()
    # get list of all input images to process
    imageDirectory = root + os.path.sep + 'input'
    print('Looking for TIFF images in:\n    ', imageDirectory)

    fileList = sorted(os.listdir(imageDirectory))
    imageList = []
    for x in range(0, len(fileList)):
        f = fileList[x]
        # get tiffs/geotiffs only
        if (f.endswith('.tif') or f.endswith('.tiff')
        or f.endswith('.gtif') or f.endswith('.gtiff')
        or f.endswith('.TIF') or f.endswith('.TIFF')
        or f.endswith('.GTIF') or f.endswith('.GTIFF')):
            imageList.append(f)
    # sort list
    imageList = sorted(imageList)

    # exit program if no images are found
    if len(imageList) == 0:
        print('No TIFF images were found. Please add files and try again.')
        print('Exiting program.\n')
        exit()

    print('The following images were found:\n')

    # display names of all images in folder
    for x in range(0, len(imageList)):
        print('    ', x+1, ': ', imageList[x])
    numImagesInList = len(imageList)

    # get image number as input from user and convert to 0-based integer
    imageNumberString = input('\nTo process a single image, enter the number '
        + "of the image.\nTo process all images enter '0'.\n")

    # ensure that the input number is valid
    while (imageNumberString.isnumeric() == False
           or int(imageNumberString) > numImagesInList
           or int(imageNumberString) < 0):
        imageNumberString = input('\nThat is not a valid image number. '
                                  + 'Try again.\n')

    imageNumber = int(imageNumberString)

    if imageNumber == 0:
        includeUserInput = False
        imageStartIndex = 0
        imageFinishIndex = numImagesInList
    else:
        includeUserInput = True
        imageStartIndex = imageNumber - 1
        imageFinishIndex = imageNumber

    # process each image in sequence
    for x in range(imageStartIndex, imageFinishIndex):

        # get image name and print
        imageFileNameWithExtension = imageList[x]

        processImage(imageDirectory, imageFileNameWithExtension,
                     includeUserInput)

################################################################################

# this conditional check allows functions to be executed directly from command
# line by passing in the function name as the first argument
#     e.g. 'python CMS_tool.py run' will execute the 'run()' function
if __name__ == '__main__':
    try:
        globals()[sys.argv[1]]()
    except:
        # if no command-line arguments are provided or if they are not valid
        # function names then simply run the tool
        run()
