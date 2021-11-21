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
    # kernel = np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]], np.float32) #maybe
    # kernel = np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]], np.float32)
    # kernel = np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]], np.float32)
    # kernel = np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]], np.float32)  # 这个可以
    dst = cv.filter2D(src, -1, kernel=kernel)
    return dst


src = cv.imread("D:/study/opencv/detection/1669800.jpg")
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)


blur = sharpen(src)
# for i in range(2):
#     blur = cv.add(blur, blur)
cv.namedWindow("blur", cv.WINDOW_NORMAL)
cv.imshow("blur", blur)


def contrast_brightness(image, c, b):
    h, w, ch = image.shape
    blank = np.zeros([h, w, ch], image.dtype)

    dst = cv.addWeighted(image, c, blank, 1-c, b)
    return dst


blur = contrast_brightness(blur, 2, 5)
cv.namedWindow("blur", cv.WINDOW_NORMAL)
cv.imshow("blur", blur)


binary = cv.medianBlur(blur, 3)
cv.namedWindow("binary", cv.WINDOW_NORMAL)
cv.imshow("binary", binary)


kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 1))
binary = cv.morphologyEx(blur, cv.MORPH_OPEN, kernel)
cv.namedWindow("binary", cv.WINDOW_NORMAL)
cv.imshow("binary", binary)


ok, th = cv.threshold(binary, 30, 255, type=cv.THRESH_BINARY)
cv.imshow("th", th)

kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
binary = cv.morphologyEx(blur, cv.MORPH_OPEN, kernel)
cv.namedWindow("binary", cv.WINDOW_NORMAL)
cv.imshow("binary", binary)


while True:
    c = cv.waitKey(50)
    if c == 27:
        break
cv.destroyAllWindows()
