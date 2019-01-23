import cv2
img = cv2.imread('sourse\lena.jpg',0)
cv2.imshow('lena2',img)
# cv2.waitKey(0)
cv2.imwrite('lena_gray.jpg',img)

im2 = cv2.imread('lena_gray.jpg',1)
cv2.imshow('lena333',im2)
cv2.waitKey(0)
