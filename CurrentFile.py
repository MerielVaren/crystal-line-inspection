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
    global src
    src = cv.imread(path)
    # 源图片展示
    # show("input image", src)
    dst = src

    '''灰度化，降色彩通道为1'''
    dst = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    # mask = numpy.zeros(src.shape)  # 黑色掩膜
    mask = np.ones(src.shape)  # 白色掩膜

    '''增强对比度'''
    dst = contrast_boost_add(dst, 2)
    show("zengqiang", dst)
    # dst = cv.medianBlur(dst, 5)
    dst = cv.GaussianBlur(dst, (5, 5), 3)

    # show("duibiduzengqiang", dst)

    '''RIO·提取局部并进行处理，中间包括对比度增强与二值化'''
    # dst = Region_One_process(dst, 5, 5, contrast_boost_in)
    dst = cv.adaptiveThreshold(
        dst, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 101, -1)
    # show("erzhihuachuli", dst)

    '''降噪处理·滤波操作'''
    dst = cv.medianBlur(dst, 13)
    # show("gaosilvbo", dst)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dst = cv.morphologyEx(dst, cv.MORPH_OPEN, kernel=kernel)  # 开操作降噪
    # show("kaicaozuo", dst)
    dst = cv.filter2D(dst, -1, kernel=kernel)
    dst = cv.normalize(dst, dst=mask, alpha=0, beta=255,
                       norm_type=cv.NORM_MINMAX)

    # '''Canny检测'''
    # edges = cv.Canny(dst, 10, 1000)
    # show("edges", edges)

    def houghP(src, dst, minLineLength):
        '''累计概率霍夫'''
        # line_lst = []
        edges = cv.Canny(dst, 10, 1000)
        background = src.copy()
        lines = cv.HoughLinesP(edges, 0.6, np.pi / 180, threshold=minLineLength,
                               minLineLength=minLineLength, maxLineGap=10)
        for x1, y1, x2, y2 in lines[:, 0]:
            line = cv.line(background, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # line_lst.append(line)
        return background

    def hough(src, dst):
        dst = delate_then_erode(dst, 20, 6, 3)
        show("erode", dst)
        res = cv.Canny(dst, 10, 1000)
        lines = cv.HoughLines(res, 1, np.pi/180, 50)
        background = src.copy()
        for line in lines:
            rho = line[0][0]  # 第一个元素是距离rho
            theta = line[0][1]  # 第二个元素是角度theta
            if(3 > theta > 2.984):
                # 该直线与第一行的交点
                pt1 = (int(rho / np.cos(theta)), 0)
                # 该直线与最后一行的焦点
                pt2 = (
                    int((rho - background.shape[0] * np.sin(theta)) / np.cos(theta)), background.shape[0])
                # 绘制一条白线
                if(pt1[0] > 50 and pt1[0] < 500):
                    cv.line(background, pt1, pt2, (255, 255, 255), 3)
        return background

    houghP_img = houghP(
        np.zeros([src.shape[0], src.shape[1], 3], src.dtype), dst, 40)
    hough_img = hough(
        np.zeros([src.shape[0], src.shape[1], 3], src.dtype), dst)

    # return dst
    return cv.add(hough_img, houghP_img)


show("img", process("D:/study/opencv/detection/1460000.bmp"))

while True:
    c = cv.waitKey(50)
    if c == 27:
        break
cv.destroyAllWindows()
