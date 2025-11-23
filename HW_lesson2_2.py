import cv2

image = cv2.imread('images/email.jpg')
image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.Canny(image, 50, 150)

cv2.imwrite('images/resultemail.jpg', image)
cv2.imshow('image', image)

cv2.waitKey(0)
cv2.destroyAllWindows()
