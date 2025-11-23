import cv2

image = cv2.imread('images/images.jpeg')
image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.Canny(image, 100, 200)

cv2.imwrite('images/resultportret.jpg', image)
cv2.imshow('image', image)

cv2.waitKey(0)
cv2.destroyAllWindows()
