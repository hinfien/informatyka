import cv2
import numpy as np
from numpy.ma.core import filled

img = np.zeros((512, 512, 3), np.uint8)

center1 = img.shape[1] // 2
center2 = img.shape[0] // 2

cv2.line(img, (0, center2), (img.shape[1], center2), (0, 0, 255), thickness=2)

cv2.line(img, (center1, 0), (center1, img.shape[0]), (0, 0, 255), thickness=2)

cv2.circle(img, (center1, center2), 50, (0, 0, 255), thickness=2)

cv2.putText(img, "Giga Vadik", (center1 + 60, center2 - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


cv2.imshow('image', img)

cv2.waitKey(0)
cv2.destroyAllWindows()