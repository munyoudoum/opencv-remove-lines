import cv2
import numpy as np

# Read in image, grayscale, and Otsu's threshold
image = cv2.imread('/Users/munyoudoum/Downloads/C10_img.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Create diagonal kernel
kernel = np.array([[0, 0, 1],
                   [0, 1, 0],
                   [1, 0, 0]], dtype=np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Find contours and filter using contour area to remove noise
cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    area = cv2.contourArea(c)
    if area < 100:
        cv2.drawContours(opening, [c], -1, (0,0,0), -1)
    else:
        cv2.putText(opening, "Area: {}".format(area), (c[0][0][0], c[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    # else:
    # cv2.drawContours(opening, [c], -1, (0,255,255), -1)

# Bitwise-xor with original image
opening = cv2.merge([opening, opening, opening])
result = cv2.bitwise_xor(image, opening)
cv2.imshow('image', image)
cv2.imshow('thresh', thresh)
cv2.imshow('opening', opening)
cv2.imshow('result', result)
cv2.waitKey()