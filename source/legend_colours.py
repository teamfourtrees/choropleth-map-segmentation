"""This script reads the following image files from the 'legend' folder:
    'mature timber.png'
    'immature timber.png'
    'not restocked.png'
    'non-commercial stands.png'
    'non-forested land.png'
    'water.png'
    'boundary.png'
and prints their RGB and HSV values to a file 'legend/legendColours.txt' for
ease of reference by an end-user.
"""

import numpy as np
import cv2
import os

# get the directory of this python script file on the local machine
# from https://stackoverflow.com/questions/5137497/find-current-directory-and-files-directory
dirPath = os.path.dirname(os.path.realpath(__file__))

# go up one folder from source file using string split and set as root folder
root,tail = os.path.split(dirPath)

# set path for legend labels text file
legendFolder = root + os.path.sep + "legend"
filePath = legendFolder+ os.path.sep + "legendColours.txt"

# delete existing legend file so a fresh copy can be made
if os.path.exists(filePath):
    os.remove(filePath)

# analyze legend colour files to extract pixel colour data and print values to file
def printColours(imageFile,nominalColour):
    # extract average RGB and gray values of image
    imgBGR = cv2.imread(legendFolder + os.path.sep + imageFile)
    B = int(np.mean(imgBGR[:,:,0]))
    G = int(np.mean(imgBGR[:,:,1]))
    R = int(np.mean(imgBGR[:,:,2]))
    gray = int((R+G+B)/3)

    # extract average HSV values of image
    imgHSV = imageHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)
    H = int(np.mean(imgHSV[:,:,0]))
    S = int(np.mean(imgHSV[:,:,1]))
    V = int(np.mean(imgHSV[:,:,2]))

    # write intensity values to file (a+ means append to file or create one if none exist)
    with open(filePath,"a+") as f:
        f.write(imageFile + "\n")
        f.write("    colour = " + nominalColour + "\n")
        f.write("    HSV = [" + str(H) + " " + str(S) + " " + str(V) + "]\n")
        f.write("    RGB = [" + str(R) + " " + str(G) + " " + str(B) + "]\n")
        f.write("    GRAY = " + str(gray) + "\n\n")

printColours("mature timber.png",           "green")
printColours("immature timber.png",         "dark yellow")
printColours("not restocked.png",           "light yellow")
printColours("non-commercial stands.png",   "dark blue")
printColours("non-forested land.png",       "off-white")
printColours("water.png",                   "light blue")
printColours("boundary.png",                "dark grey")

print("The legend colours have been saved to file in:")
print("    ",filePath,"\n")
