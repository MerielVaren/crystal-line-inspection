import cv2 as cv
import numpy as np


def sharpen(src):
    blured = cv.blur(src, (1, 80))  # 横向和纵向的模糊程度
    cv.imshow("blured", blured)
    # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    # kernel = np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]], np.float32)
    # kernel = np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]], np.float32)
    # kernel = np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]], np.float32)
    kernel = np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]],
                      np.float32)  # 这个可以
    # kernel = np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]], np.float32)
    # kernel = np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]], np.float32)
    # kernel = np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]], np.float32)
    # kernel = np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]], np.float32)
    dst = cv.filter2D(src, -1, kernel=kernel)
    cv.namedWindow("ruihua", cv.WINDOW_NORMAL)
    cv.imshow("ruihua", dst)
    return dst


src = cv.imread("D:/study/opencv/detection/1668600.jpg")
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)


# blur = sharpen(src)
# for i in range(2):
#     blur = cv.add(blur, blur)
#     cv.namedWindow("blur", cv.WINDOW_NORMAL)
# cv.imshow("blur", blur)


ath = cv.adaptiveThreshold(
    gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, -1)
# cv.namedWindow("ath", cv.WINDOW_NORMAL)
# cv.imshow("ath", ath)

ret, th = cv.threshold(gray, 18, 255, cv.THRESH_BINARY)  # 20
# cv.namedWindow("th", cv.WINDOW_NORMAL)
# cv.imshow("th", th)

ior = cv.bitwise_or(ath, th)
# cv.namedWindow("ior", cv.WINDOW_NORMAL)
# cv.imshow("ior", ior)

# 降噪
# dst = cv.blur(ath, (5, 5))
# cv.imshow("dst", ath)

kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 1))
binary = cv.morphologyEx(ior, cv.MORPH_OPEN, kernel)
# cv.namedWindow("binary", cv.WINDOW_NORMAL)
# cv.imshow("binary", binary)

# ret, th = cv.threshold(
#     gray, 10, 255, cv.THRESH_BINARY)

'''
方案1
'''
# image = binary.copy()
# kernele = cv.getStructuringElement(cv.MORPH_RECT, (3, 1))
# dst = cv.erode(image, kernele)
# cv.namedWindow("ig", cv.WINDOW_NORMAL)
# cv.imshow("ig", dst)

'''
方案2
'''
kerneld = cv.getStructuringElement(cv.MORPH_RECT, (1, 10))
dst = cv.dilate(binary, kerneld)
# cv.imshow("dst1", dst)
kernele = cv.getStructuringElement(cv.MORPH_RECT, (12, 1))
dst = cv.erode(dst, kernele)

kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 1))
dst = cv.morphologyEx(dst, cv.MORPH_OPEN, kernel)
cv.namedWindow("dst2", cv.WINDOW_NORMAL)
# dst = cv.bitwise_not(dst)
cv.imshow("dst2", dst)


minLineLength = 20
edges = cv.Canny(dst, 10, 1000)
cv.imshow("edges", edges)
hough = src.copy()
lines = cv.HoughLinesP(
    edges, 0.6, np.pi / 180, threshold=minLineLength, minLineLength=minLineLength, maxLineGap=6)
for x1, y1, x2, y2 in lines[:, 0]:
    cv.line(hough, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv.namedWindow("hough", cv.WINDOW_NORMAL)
cv.imshow("hough", hough)  # 左右设置阈值差

while True:
    c = cv.waitKey(50)
    if c == 27:
        break
cv.destroyAllWindows()
