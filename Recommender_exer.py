import numpy as np
from math import *
from PIL import Image
from Recommender import * 

def convert2BW(img_in):
    # routine reads in the image from file
    # and converts it to black-and-white

    # open the image file
    im_file = Image.open(img_in)
    # convert image to monochrome
    im_BW = im_file.convert('L')
    # convert to num-array
    im_array = np.array(im_BW)

    # convert back to image (for later use)
    # im_file2 = Image.fromarray(im_array)
    # show image (for later use)
    # im_file2.show()

    return im_array

if __name__ == '__main__':

    # matrix completion problem (image denoising)

    # read in a very noisy image
    a = convert2BW('PhotoNoise.png')
    (na,ma) = a.shape

    # take a look at the noisy image
    im_ein = Image.fromarray(a.astype(np.uint8))
    # im_ein.show()

    # define as mask all white pixels
    for i in range(na):
        for j in range(ma):
            if a[i][j] == 255:
                a[i][j] = False #float('NaN')

    # initialize parameters
    fratio = .8 
    mu  = 1./np.linalg.norm(a,2)
    rho = 1.2172 + 1.8588*fratio

    # call IALM-algorithm
    AA,EE = IALM(a,mu,rho)

    # show the restored image
    im_ein = Image.fromarray(AA.astype(np.uint8))
    im_ein.show()

