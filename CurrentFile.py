import cv2 as cv
import numpy as np


def contrast_brightness(image, c, b):
    h, w = image.shape
    blank = np.zeros([h, w], image.dtype)
    dst = cv.addWeighted(image, c, blank, 1-c, b)
    return dst


def delate_then_erode(img, times, dilate, erode):
    dst = img
    for i in range(times):
        kerneld = cv.getStructuringElement(cv.MORPH_RECT, (1, dilate))
        dst = cv.dilate(dst, kerneld)
        kernele = cv.getStructuringElement(cv.MORPH_RECT, (erode, 1))
        dst = cv.erode(dst, kernele)
    return dst


def show(name, image):
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.imshow(name, image)


def contrast_boost_add(image, n=3):
    dst = image
    m = dst
    for i in range(n):
        dst = cv.add(m, dst)
    return dst


def process(path):
    # 将路径的照片文件转化成多维数组
    src = cv.imread(path)
    # 源图片展示
    # show("input image", src)
    dst = src

    # show("temp", dst)

    '''灰度化，降色彩通道为1'''
    dst = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    # mask = numpy.zeros(src.shape)  # 黑色掩膜
    mask = np.ones(src.shape)  # 白色掩膜

    '''增强对比度'''
    dst = contrast_boost_add(dst, 2)
    show("zengqiang", dst)
    # dst = cv.medianBlur(dst, 5)
    dst = cv.GaussianBlur(dst, (5, 5), 3)

    show("duibiduzengqiang", dst)

    '''RIO·提取局部并进行处理，中间包括对比度增强与二值化'''
    # dst = Region_One_process(dst, 5, 5, contrast_boost_in)
    dst = cv.adaptiveThreshold(
        dst, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 101, -1)
    show("erzhihuachuli", dst)

    '''降噪处理·滤波操作'''
    dst = cv.medianBlur(dst, 13)
    show("gaosilvbo", dst)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dst = cv.morphologyEx(dst, cv.MORPH_OPEN, kernel=kernel)  # 开操作降噪
    show("kaicaozuo", dst)
    dst = cv.filter2D(dst, -1, kernel=kernel)
    dst = cv.normalize(dst, dst=mask, alpha=0, beta=255,
                       norm_type=cv.NORM_MINMAX)

    # dst = delate_then_erode(dst, 20, 6, 3)

    # '''Canny检测'''
    # edges = cv.Canny(dst, 10, 1000)
    # show("edges", edges)

    '''累计概率霍夫'''
    minLineLength = 40
    line_lst = []
    hough = src.copy()
    edges = cv.Canny(dst, 10, 1000)
    show("edges", edges)
    lines = cv.HoughLinesP(edges, 0.6, np.pi / 180, threshold=minLineLength,
                           minLineLength=minLineLength, maxLineGap=10)
    for x1, y1, x2, y2 in lines[:, 0]:
        line = cv.line(hough, (x1, y1), (x2, y2), (0, 255, 0), 2)
        line_lst.append(line)
    cv.namedWindow("hough", cv.WINDOW_NORMAL)
    show("hough", hough)

    return dst


show("img", process("D:/study/opencv/detection/1462000.bmp"))

while True:
    c = cv.waitKey(50)
    if c == 27:
        break
cv.destroyAllWindows()
