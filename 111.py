import cv2

canny_lthreshold = 50  # Canny edge detection low threshold
canny_hthreshold = 150  # Canny edge detection high threshold
img = cv2.imread('sourse/lena.jpg')
print(img.shape)
# gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  #图像转换为灰度图
# print(gray.shape)
line_img = cv2.Canny(img, canny_lthreshold, canny_hthreshold)
print(line_img.shape)
res_img = cv2.addWeighted(img, 0.8, line_img, 1, 0)  # 将处理后的图像与原图做融合,
cv2.imshow('result',res_img)
cv2.waitKey(0)