# https://stackoverflow.com/questions/60780831/python-how-to-cut-out-an-area-with-specific-color-from-image-opencv-numpy
import glob
import numpy as np
from PIL import Image, ImageFilter
import cv2


for filename in glob.glob('/Users/munyoudoum/Desktop/compute_vision/remove_lines/plots/**/*.png', recursive=True):
    # Open image and make into Numpy array
    im = Image.open(filename).convert('RGB')
    na = np.array(im)
    orig = na.copy()    # Save original

    # Median filter to remove outliers
    im = im.filter(ImageFilter.MedianFilter(3))

    # Find X,Y coordinates of all yellow pixels
    yellowY, yellowX = np.where(np.all(na==[229,236,246],axis=2))

    top, bottom = yellowY[0], yellowY[-1]
    left, right = yellowX[0], yellowX[-1]
    print(top,bottom,left,right)

    # Extract Region of Interest from unblurred original
    orig= cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    ROI = orig[top:bottom, left:right]
    cv2.imshow("original", orig)
    cv2.imshow("ROI", ROI)
    cv2.waitKey(0)
    filename = "_".join(filename.rsplit(".",1)[0].rsplit("/")[-2:])
    # Image.fromarray(ROI).save(f'/Users/munyoudoum/Desktop/compute_vision/remove_lines/cropped/cropped-{filename}.png')