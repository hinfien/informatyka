import cv2
import numpy as np

img = cv2.imread('images/WIN_20260214_19_07_16_Pro.jpg')

cv2.rectangle(img, (455, 250), (705, 540), (0, 255, 0), 2)

cv2.putText(img, "Vadym Maslov", (455, 590), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cv2.imwrite('HW_lesson_3.jpg', img)

cv2.imshow('Giga dz', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
