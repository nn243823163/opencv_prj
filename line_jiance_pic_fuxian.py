#coding：utf-8
#自己复现车道检测算法 line_

import matplotlib.image as mplimg
import numpy as np
import cv2

blur_ksize = 5  # Gaussian blur kernel size 这个参数该怎么设置？
canny_lthreshold = 50  # Canny edge detection low threshold
canny_hthreshold = 150  # Canny edge detection high threshold

# Hough transform parameters
rho = 1 #rho的步长，即直线到图像原点(0,0)点的距离
theta = np.pi / 18  #theta的范围
threshold = 10 #累加器中的值高于它时才认为是一条直线
min_line_length = 60 #线的最短长度，比这个短的都被忽略
max_line_gap = 50 #两条直线之间的最大间隔，小于此值，认为是一条直线, 参数越大，直线越多

def roi_mask(img, vertices): #img是输入的图像，verticess是兴趣区的四个点的坐标（三维的数组）
  mask = np.zeros_like(img) #生成与输入图像相同大小的图像，并使用0填充,图像为黑色
  # cv2.imshow('111',mask)
  # cv2.waitKey(0)
  #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
  if len(img.shape) > 2:
    channel_count = img.shape[2]   # i.e. 3 or 4 depending on your image
    mask_color = (255,) * channel_count #如果 channel_count=3,则为(255,255,255)
  else:
    mask_color = 255
  cv2.fillPoly(mask, vertices, mask_color)#使用白色填充多边形，形成蒙板
  # cv2.imshow('111',mask)
  # cv2.waitKey(0)
  masked_img = cv2.bitwise_and(img, mask)#img&mask，经过此操作后，兴趣区域以外的部分被蒙住了，只留下兴趣区域的图像
  return masked_img


def draw_lines(img, lines, color=[255, 255, 0], thickness=2):
  for line in lines:
    for x1, y1, x2, y2 in line:
      cv2.line(img, (x1, y1), (x2, y2), color, thickness)
  # cv2.imshow('111',img)
  # cv2.waitKey(0)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
  lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)#函数输出的直接就是一组直线点的坐标位置（每条直线用两个点表示[x1,y1],[x2,y2]）

  line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)#生成绘制直线的绘图板，黑底
  # cv2.imshow('111',line_img)
  # cv2.waitKey(0)
  # draw_lines(line_img, lines)
  draw_lines(line_img, lines)
  cv2.imshow('lines', line_img)
  # cv2.waitKey(0)
  return line_img

img = cv2.imread('sourse/line1.jpg')
cv2.imshow('original',img)
# cv2.waitKey(0)
# print(img.shape)  # shape: 540, 960, 3

#目标区域的四个点坐标，roi_vtx是一个三维的数组,img上的坐标是怎么样的？这个区域是下方的区域吗？
roi_vtx = np.array([[(0, img.shape[0]), (460, 325), (520, 325), (img.shape[1], img.shape[0])]])
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  #图像转换为灰度图
# cv2.imshow('111',gray)
# cv2.waitKey(0)
blur_gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0, 0)  # 使用高斯模糊去噪声，高斯去燥原理？
# cv2.imshow('112',blur_gray)
# cv2.waitKey(0)
edges = cv2.Canny(blur_gray, canny_lthreshold, canny_hthreshold)#使用Canny进行边缘检测，边缘检测原理
# print(edges.shape)  #540,960
# cv2.imshow('canny',edges)
# cv2.waitKey(0)
roi_edges = roi_mask(edges, roi_vtx)  # 对边缘检测的图像生成图像蒙板，去掉不感兴趣的区域，保留兴趣区
cv2.imshow('roi_edges',roi_edges)
# cv2.waitKey(0)

# 使用霍夫直线检测，并且绘制直线
line_img = hough_lines(roi_edges, rho, theta, threshold, min_line_length, max_line_gap)
# print(line_img.shape) #540,960,3
res_img = cv2.addWeighted(img, 0.8, line_img, 1, 0)  # 将处理后的图像与原图做融合，两个图片必须形状相同
cv2.imshow('result',res_img)
cv2.waitKey(0)