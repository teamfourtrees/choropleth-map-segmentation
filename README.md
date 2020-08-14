# Choropleth Map Segmentation Tool

### This project contains source code for a command-line tool that performs image segmentation and file conversion on scanned geoTIFF choropleth maps.  
#### Developed for the NFIS team at the Pacific Forestry Centre (Natural Resources Canada) as part of a Camosun College ICS Capstone Project, May-August 2020.

Please review the enclosed user guide (CMS_Tool_User_Guide.pdf) for instructions on installing and using the Choropleth Map Segmentation Tool.

#### All files relating to the CMS Tool are stored in the choropleth-map-segmentation folder, which contains the following items:

FILES:
- `CMS_Tool_User_Guide.pdf` – Contains step-by-step instructions for installing and using the CMS Tool.
- `README.md` – Contains a brief overview of this project.

SUBFOLDERS:
- `config` – Contains configuration details for installing the anaconda environment and for styling shapefiles.
- `input` – Stores geoTIFF images placed by the user for processing.
- `legend` – Contains images from the original map legend used for analysis of pixel categories (see the legend_colours.py script for details).
- `output` – Stores output for each image processed by the CMS Tool. Subfolders within each image folder store the output files based on format.
- `source` – Contains source code for the CMS Tool (CMS_tool.py) and a script to analyze map legend pixels (legend_colours.py).


##### Please raise an issue if you have questions or encounter problems with the tool. Thanks!
