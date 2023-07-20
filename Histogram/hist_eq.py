import os
from PIL import Image
from numpy import *


def histeq(im,nbr_bins=256):
  """  Histogram equalization of a grayscale image. """

  # get image histogram
  imhist,bins = histogram(im.flatten(),nbr_bins,normed=True)
  cdf = imhist.cumsum() # cumulative distribution function
  cdf = 255 * cdf / cdf[-1] # normalize

  # use linear interpolation of cdf to find new pixel values
  im2 = interp(im.flatten(),bins[:-1],cdf)
  return im2.reshape(im.shape), cdf


abs_path = __file__
im = array(Image.open(os.path.join(abs_path[:abs_path.rfind('/')],'../Images/rocket.jpeg')).convert('L'))
im2,cdf = histeq(im)