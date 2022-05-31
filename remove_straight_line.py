import cv2
import numpy as np
import glob

DEBUG = 1

for filename in glob.glob('/Users/munyoudoum/Desktop/compute_vision/remove_lines/plots/**/*.png', recursive=True):
    print(filename)
    # Read in image, grayscale, and Otsu's threshold
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Create diagonal kernel
    kernel = np.array([[0, 0, 1],
                    [0, 1, 0],
                    [1, 0, 0]], dtype=np.uint8)
    opening2 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    kernel = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]], dtype=np.uint8)
    opening = cv2.morphologyEx(opening2, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours and filter using contour area to remove noise
    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    working_area = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 90:
            cv2.drawContours(opening, [c], -1, (0,0,0), -1)
        else:
            cv2.putText(opening, "Area: {}".format(area), (c[0][0][0], c[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            working_area += area
        # else:
        # cv2.drawContours(opening, [c], -1, (0,255,255), -1)

    cv2.putText(opening, "Working Area: {}".format(working_area), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    cv2.putText(opening, "Total Area: {}".format(image.shape[0]*image.shape[1]), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    cv2.putText(opening, "Ratio: {}".format(working_area/(image.shape[0]*image.shape[1])*100), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    # draw line in 10% of x axis
    cv2.line(opening, (int(image.shape[1]*0.08), 0), (int(image.shape[1]*0.08), image.shape[0]), (255,255,255), 2)
    # draw a horizontal line in 11% of y axis
    cv2.line(opening, (0, int(image.shape[0]*0.11)), (image.shape[1], int(image.shape[0]*0.11)), (255,255,255), 2)
    # draw a horizontal line in 85% of y axis
    cv2.line(opening, (0, int(image.shape[0]*0.85)), (image.shape[1], int(image.shape[0]*0.85)), (255,255,255), 2)
    # draw vertical line in 92% of x axis
    cv2.line(opening, (int(image.shape[1]*0.92), 0), (int(image.shape[1]*0.92), image.shape[0]), (255,255,255), 2)

    # Bitwise-xor with original image
    opening = cv2.merge([opening, opening, opening])
    result = cv2.bitwise_xor(image, opening)
    if DEBUG:
        cv2.imshow('image', image)
        cv2.imshow('thresh', thresh)
        cv2.imshow('opening', opening)
        cv2.imshow('result', result)
        cv2.waitKey()
    filename = "_".join(filename.rsplit(".",1)[0].rsplit("/")[-2:])
    # cv2.imwrite("/Users/munyoudoum/Desktop/compute_vision/remove_lines/processed_images/"+filename+"_test.png", result)