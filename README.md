# choropleth-map-segmentation
### Source code for a command-line utility that performs image segmentation and file conversion on scanned geoTIFF choropleth maps.  Developed for the Pacific Forestry Centre as part of the Camosun College ICS Capstone Project, May-August 2020.

### Installation Instructions:

1. First, install Anaconda and Python 3 (included with Anaconda) if this is not already present on your machine. Instructions for this are available from anaconda.org

2. Next, create a new Anaconda (conda) environment. Using the command-line interface, enter the command:
`conda create -n mapEnv opencv pillow numpy matplotlib scipy rasterio fiona`
  - Note: <mapEnv> is an example, and can be replaced with the environment name of your choice.

3. Clone this git repository into a directory on your machine where you have access permissions to run commands.

4. Collect the geoTIFF choropleth maps for processing.
  - If using command line only, the command `wget` will work on unix-based systems to retrieve data from a given URL.

5. Copy the geoTIFFs to the 'input' folder

TODO: Finalize procedure
