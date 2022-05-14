import cv2
import numpy as np

image = cv2.imread('/Users/munyoudoum/Downloads/C6_img.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)
edged = cv2.Canny(gray, 50, 200)

lines = cv2.HoughLines(edged, 1, np.pi/180, 90)

height, width = image.shape[:2]
for x, i in enumerate((gray, edged, image)):
    cv2.imshow(str(x), i)
cv2.waitKey(0)
mask = np.zeros((height,width), np.uint8)

for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(mask,(x1,y1),(x2,y2),(255,255,255),3)

output = cv2.inpaint(image, mask, 3, flags=cv2.INPAINT_NS)

cv2.imshow("Output", output)
cv2.waitKey(0)