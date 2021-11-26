# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 10:39:46 2021

@author: 12406
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils

img=cv2.imread("D:\\pictureprocessing\\1669800.jpg",0)
# img = cv2.GaussianBlur(img, (3,3), 0)


# img = 2 * img
# # 高斯平滑
# img = cv2.GaussianBlur(img,(3,3),0)
# edges = cv2.Canny(img, 150, 100, apertureSize = 3)
# lines = cv2.HoughLines(edges,1,np.pi/180,120) 
# result = img.copy()
# for line in lines:
#     rho = line[0][0]
#     theta= line[0][1]
#     if  (theta < (np.pi/4. )) or (theta > (3.*np.pi/4.0)):# 垂直线
#         #与第一行交点
#         pt1 = (int(rho/np.cos(theta)),0)
#         #与最后一行交点
#         pt2 = (int((rho-result.shape[0]*np.sin(theta))/np.cos(theta)),result.shape[0])
#         cv2.line( result, pt1, pt2, (255))
#     else:# 水平线
#         # 该直线与第一列的交点
#         pt1 = (0,int(rho/np.sin(theta)))
#         #与最后一列交点
#         pt2 = (result.shape[1], int((rho-result.shape[1]*np.cos(theta))/np.sin(theta)))
#         cv2.line(result, pt1, pt2, (255), 1)
 
# cv2.imshow('Canny', edges )
# cv2.imshow('Result', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#Sobel算子
# img = 5 * img
# #x和y方向上差分
# x = cv2.Sobel(img, -1, 1, 0, ksize=3)
# y = cv2.Sobel(img, -1, 0, 1, ksize=3)
# absX = cv2.convertScaleAbs(x)   # 转回uint8
# absY = cv2.convertScaleAbs(y)
# #叠加两个方向的处理后的图片
# dst = cv2.addWeighted(absX,0.5,absY,0.5,0)
# cv2.imshow("Result", dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



#拉普拉斯算子
# out = 2 * img
# gray_lap = cv2.Laplacian(out,cv2.CV_16S,ksize = 5)
# dst = cv2.convertScaleAbs(gray_lap)
# # dst = cv2.GaussianBlur(dst, (3,3), 0)
# # dst = cv2.medianBlur(dst,1)
# cv2.imshow('result',dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# minThres = 6
# # 读取图像
# img1 = cv2.imread('D:\\pictureprocessing\\1669800.jpg')
# img2 = cv2.imread('D:\\pictureprocessing\\1669800.jpg')
# # 中值滤波
# img1 = cv2.medianBlur(img1, 25)
# # 图像差分
# diff = cv2.absdiff(img1, img2)
# cv2.imshow('diff', diff)  # 结果图
# gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
# # 二值化
# _, thres = cv2.threshold(gray, minThres, 255, cv2.THRESH_BINARY)
# cv2.imshow('thres', thres)
# # 查找轮廓
# thres, contours, hierarchy = cv2.findContours(thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# # 输出轮廓个数
# print(len(contours))
# for i in range(0, len(contours)):
#     length = cv2.arcLength(contours[i], True)
#     # 通过轮廓长度筛选
#     if length > 5:
#         cv2.drawContours(img2, contours[i], -1, (0, 0, 255), 2)
# cv2.imshow('result', img2)  # 结果图
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# 自适应均衡化
# cv2.imshow("src", img)
# # 直方图均衡化
# dst1 = cv2.equalizeHist(img)
# cv2.imshow("dst1", dst1)
# # 自适应直方图均衡化
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# dst2 = clahe.apply(img)
# cv2.imshow("dst2", dst2)
# cv2.waitKey(0)



# 自适应二值化
# th1 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,7,2)
# th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,2)
# cv2.imshow("th1",th1)
# cv2.imshow("th2",th2)
# cv2.waitKey(0)
