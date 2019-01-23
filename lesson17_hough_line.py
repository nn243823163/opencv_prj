#coding:utf-8
import cv2
import numpy as np

img = cv2.imread('sourse\line1.jpg')
# cv2.imshow('aa1',img)
# cv2.waitKey(0)
# print(img.shape)
drawing = np.zeros(img.shape[:], dtype=np.uint8)
# print(drawing.shape)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('aa2',gray)
# cv2.waitKey(0)
# print(gray.shape)
edges = cv2.Canny(gray,50,150)      ####这个语句不懂？
# print(edges.shape)
# cv2.imshow('aa',edges)
# cv2.waitKey(0)

lines = cv2.HoughLines(edges, 0.8, np.pi / 180, 90)
# print(lines)

for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * a )
    cv2.line(drawing,(x1,y1),(x2,y2),(0,0,255))

cv2.imshow('111',drawing)
cv2.waitKey(0)


