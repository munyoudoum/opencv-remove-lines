import diplib as dip
import matplotlib.pyplot as pp

img = 1 - pp.imread('/Users/munyoudoum/Downloads/C10_img.png')
lines = dip.PathOpening(img, length=300, mode={'constrained'})
text = img - lines
text = dip.AreaOpening(text, filterSize=5)
lines = lines > 0.5
text = text > 0.5
lines -= dip.BinaryPropagation(text, lines, connectivity=-1, iterations=3)
img[lines] = 0