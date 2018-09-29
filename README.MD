#Limb Darkening Correction

This is an limb darkening implementation based on Bertello 2010 paper https://link.springer.com/article/10.1007/s11207-010-9570-z.

##Algorithm:

1. Idea is to calculate solar disc radius and centre.
2. Divide the solar disc by the median filtered image.

##Method Followed

1. A gradient Image is calculated, using the method described [here](http://www.astro.ucla.edu/~ulrich/MW_SPADP/Fits_Info.html)

2. While following the method, it has been observed that, at very high slopes, It is impractical to make 7x5 boxes before and after the point, hence a parameter called angle is introduced, which considers all points in the angular area -angle to +angle with slope infinite. This value of angle depends on instrument to instrument.

3. For Kodaikanal data, I found the value of angle values 25 for H alpha data and 7 for CaK data.

4. Histogram is found for the gradient image (which is basically an edge enhanced image), and using the histogram, a threshold value has been obtained, above which all points are considered on solar disc radii.

5. Using all those points, least square fitting has been done to find accurate radii and centre coordinates.

6. Once we have the centre and radius, we convert flat fielded images to 512x512 for faster processing, do a median filter with size 105x105, and divide the solar disc with median filtered image to get the limb darkening corrected image.

##How to Run the code:

###To Compile

**python3 setup.py build_ext --inplace**


###Create the following directories:
1. output_images
2. gradient_images
3. profiles

###Run this code on ipython3 shell:

import halpha

base_path = '/Users/harshmathur/Documents/H alpha20170803/'

dark_image = 'Dark_20170803_133401800.fits'

flat_image = 'Flat_20170803_112801100.fits'

angle = 7 # 25 for H alpha, 7 for CaK images

halpha.limb_darkening_corrected_flow(
    base_path,
    image,
    dark_image,
    flat_image,
    angle
)


