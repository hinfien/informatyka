import cv2
import numpy as np

img = cv2.imread("images/image5.jpg")
img = cv2.resize(img, (600, 400))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

count = 0

for cnt in contours:
    area = cv2.contourArea(cnt)
    
    if area > 500:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        count += 1

print(f"Кількість знайдених наліпок: {count}")
cv2.imshow("Nalipky", img)
cv2.imwrite("images/result.jpg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
