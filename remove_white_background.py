import glob
import numpy as np
from PIL import Image, ImageFilter

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
    ROI = orig[top:bottom, left:right]
    filename = "_".join(filename.rsplit(".",1)[0].rsplit("/")[-2:])
    Image.fromarray(ROI).save(f'/Users/munyoudoum/Desktop/compute_vision/remove_lines/cropped/cropped-{filename}.png')