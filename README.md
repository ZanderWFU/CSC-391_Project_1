# CSC-391_Project_1
Spatial and Frequency Filtering Project
Created by Zander Miller

This program is split into three files, each corresponding to a different part of the assigned project:

The Spatial_Filter.py file corresponds to the "Spatial Filter" section and has Box, Gaussian, and Median Blur functions, as well as Canny edge detection

The Frequency_Analysis.py file corresponds to the Frequency Analysis section, and can perform a 2-D DFT on the supplied image, and plot the result in 2-D or 3-D. It also has a function of zero-ing out a square or squares on the DFT, and displaying and saving the results.

The Frequency_Filtering.py file corresponds to the Frequency Filtering section, it can perform a 2-D DFT on the supplied image, and run it through an Ideal and Butterworth filter (for n = 1-4) in either low-pass or high-pass mode.

The all allow the user to control the functionality through the command line, and it should be intuitive.
