import cv2 as cv
import numpy as np
import os


def lines_crossed(line1, line2, mode=1):
    '''
    mode:
    1、直线与直线
    2、直线与线段
    '''

    if line1 == None or line2 == None:
        return False, (0, 0)
    point_is_exist = False

    x = y = 0
    # line1 = [int(i) for i in line1]
    # line2 = [int(i) for i in line2]
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    # 判断相交，第一个为直线，第二个为线段，每一个参数都含有两个点坐标
    # if mode == 2:
    #     if ((y3*(x2-x1)-((y2-y1)*(x3-x1) + y1*(x2-x1))) * (y4*(x2-x1)-((y2-y1)*(x4-x1) + y1*(x2-x1))) > 0):
    #     if ((x1 - x3) * (y2 - y3) - (y1 - y3) * (x2 - x3)) * ((x1 - x4) * (y2 - y3) - (y1 - y4) * (x2 - x4)) > 0:
    #         return False, (0, 0)

    if (x2 - x1) == 0:
        k1 = None
        b1 = 0
    else:
        k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
        b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键

    if (x4 - x3) == 0:  # L2直线斜率不存在
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在
        b2 = y3 * 1.0 - x3 * k2 * 1.0

    if k1 is None:
        if not k2 is None:
            x = x1
            y = k2 * x1 + b2
            point_is_exist = True
    elif k2 is None:
        x = x3
        y = k1 * x3 + b1
    elif not k2 == k1:
        x = (b2 - b1) * 1.0 / (k1 - k2)
        y = k1 * x * 1.0 + b1 * 1.0
        point_is_exist = True

    point = (int(x), int(y))
    if mode == 2:
        if point[0] > max(x3, x4) or point[0] < min(x3, x4):
            point_is_exist = False
            point = (0, 0)

    return point_is_exist, (int(x), int(y))


def contrast_boost_add(image, n=3):
    '''图片自加处理'''
    dst = image
    m = dst
    for i in range(n):
        dst = cv.add(m, dst)
    return dst


def contrast_brightness(image, c, b):
    '''增强对比度, c为对比度默认为1, b为亮度默认为0'''
    h, w = image.shape
    blank = np.zeros([h, w], image.dtype)
    dst = cv.addWeighted(image, c, blank, 1-c, b)
    return dst


def dilate_then_erode(img, times, dilate_kernel, erode_kernel, dilate_times=1, erode_times=1):
    '''膨胀并腐蚀
        img: 图片
        times: 总次数
        dilate_kernel: 膨胀卷积核
        erode_ kernel: 腐蚀卷积核
        dilate_times: 每一次中连续膨胀次数
    '''
    dst = img
    for i in range(times):
        for j in range(dilate_times):
            kerneld = cv.getStructuringElement(
                cv.MORPH_RECT, (1, dilate_kernel))
            dst = cv.dilate(dst, kerneld)
        for j in range(erode_times):
            kernele = cv.getStructuringElement(
                cv.MORPH_RECT, (erode_kernel, 1))
            dst = cv.erode(dst, kernele)
    return dst


def dilate_and_erode(image):

    ker_dilate = cv.getStructuringElement(cv.MORPH_RECT, (2, 3))
    ker_erode = cv.getStructuringElement(cv.MORPH_RECT, (3, 2))
    # dst = cv.erode(dst, ker_erode)
    dst = cv.dilate(image, ker_dilate)
    dst = cv.dilate(dst, ker_dilate)
    dst = cv.dilate(dst, ker_dilate)
    dst = cv.erode(dst, ker_erode)

    return dst


def show(name, image):
    '''
    展示图片
    name: 窗口名
    image: 图片
    '''
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.imshow(name, image)


def dye(image, lw=5, rw=5,  sh=3, value=255):
    '''截断染色函数，将图像的两边染色
        image: 图片
        lw: 左边染色1/lw
        rw: 右边染色1/rw
        sh: 上面的染色范围
        value: 染色的灰度值
    '''
    h, w = image.shape
    mask = np.ones([h, w], np.uint8)
    mask *= value
    mask[2*int(h/sh):h, int(w/lw): int(w-w/rw)] = 0
    dst = cv.add(mask, image)
    return dst


def houghP(src, dst, minLineLength):
    '''累计概率霍夫'''
    dst = cv.medianBlur(dst, 13)

    edges = cv.Canny(dst, 10, 1000)
    background = src.copy()
    lines = cv.HoughLinesP(edges, 0.6, np.pi / 180, threshold=minLineLength,
                           minLineLength=minLineLength, maxLineGap=10)
    for x1, y1, x2, y2 in lines[:, 0]:
        cv.line(background, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return background, [(x1, y1, x2, y2) for x1, y1, x2, y2 in lines[:, 0]]


def hough(src, dst):
    '''霍夫直线检测
        src: 背景图
        dst: 检测图
        返回值:
        background: 绘制直线后的图片
        ((x1, y1, x2, y2) if len(lines) else None): 检测出的线段集合 (以 (x1, y1, x2, y2) 的形式)
    '''

    '''定义卷积核'''
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    kerneld = cv.getStructuringElement(cv.MORPH_RECT, (1, 3))
    kernele = cv.getStructuringElement(cv.MORPH_RECT, (3, 1))

    '''膨胀腐蚀'''
    dst = cv.dilate(dst, kerneld)
    dst = cv.dilate(dst, kerneld)
    dst = cv.dilate(dst, kerneld)
    dst = cv.dilate(dst, kerneld)
    dst = cv.dilate(dst, kerneld)
    dst = cv.dilate(dst, kerneld)

    dst = cv.erode(dst, kernele)
    dst = cv.erode(dst, kernele)
    dst = cv.erode(dst, kernele)

    dst = cv.dilate(dst, kerneld)
    dst = cv.dilate(dst, kerneld)
    dst = cv.dilate(dst, kerneld)
    dst = cv.dilate(dst, kerneld)
    dst = cv.dilate(dst, kerneld)
    dst = cv.dilate(dst, kerneld)
    dst = cv.dilate(dst, kerneld)
    dst = cv.dilate(dst, kerneld)
    dst = cv.dilate(dst, kerneld)
    dst = cv.dilate(dst, kerneld)

    dst = cv.erode(dst, kernele)
    dst = cv.erode(dst, kernele)
    dst = cv.erode(dst, kernele)

    dst = cv.dilate(dst, kerneld)
    dst = cv.dilate(dst, kerneld)
    dst = cv.dilate(dst, kerneld)
    dst = cv.dilate(dst, kerneld)
    dst = cv.dilate(dst, kerneld)
    dst = cv.dilate(dst, kerneld)
    dst = cv.dilate(dst, kerneld)
    dst = cv.dilate(dst, kerneld)
    dst = cv.dilate(dst, kerneld)
    dst = cv.dilate(dst, kerneld)
    dst = cv.dilate(dst, kerneld)
    dst = cv.dilate(dst, kerneld)

    dst = cv.erode(dst, kernele)
    dst = cv.erode(dst, kernele)

    dst = cv.dilate(dst, kerneld)
    dst = cv.dilate(dst, kerneld)
    dst = cv.dilate(dst, kerneld)
    dst = cv.dilate(dst, kerneld)
    dst = cv.dilate(dst, kerneld)

    dst = cv.erode(dst, kernele)
    dst = cv.erode(dst, kernele)
    dst = cv.erode(dst, kernele)

    '''resize函数具有抗齿距的效果'''
    dst = cv.resize(dst, dst.shape, interpolation=cv.INTER_CUBIC)
    dst = cv.resize(dst, dst.shape, interpolation=cv.INTER_CUBIC)

    # show("temp", dst)

    dst = np.array(dst, dtype=np.uint8)
    # 形态学闭操作
    res = cv.Canny(dst, 10, 1000)
    # show("canny", res)
    lines = cv.HoughLines(res, rho=2, theta=np.pi/180, threshold=85)
    # 筛选并排序
    lines = sorted([line for line in lines if 2.88 <
                    line[0][1] < 3.4], key=lambda x: x[0][1])
    background = src.copy()

    if len(lines):
        # 由极坐标系转换成的霍夫空间下的横纵坐标
        line = lines[len(lines)//2]
        rho = line[0][0]  # 第一个元素是距离rho
        theta = line[0][1]  # 第二个元素是角度theta
        x1 = int(rho / np.cos(theta))
        y1 = 0
        x2 = int((rho - background.shape[0]
                  * np.sin(theta)) / np.cos(theta))
        y2 = background.shape[0]

        # 该直线与第一行的交点
        pt1 = (x1, y1)
        # 该直线与最后一行的焦点
        pt2 = (x2, y2)
        cv.line(background, pt1, pt2, (255, 0, 0), 5)

    return background, ((x1, y1, x2, y2) if len(lines) else None)


def process(img_path, save_path):
    '''----------------------------------前期操作-----------------------------------'''

    '''创建文件夹'''
    try:
        os.mkdir("D:/study/opencv/test/" + img_path[-11:-4])
    except:
        pass

    '''展示图全局化'''
    global src
    src = cv.imread(img_path)

    '''----------------------------------图像预处理-----------------------------------'''

    '''灰度化，降色彩通道为1'''
    dst = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    mask = np.ones(src.shape)  # 白色掩膜

    '''增强对比度'''
    dst = contrast_boost_add(dst, 2)

    preview = dst

    '''寻找最大值和最小值'''
    minVal, maxVal, minIdx, maxIdx = cv.minMaxLoc(dst)
    # dst = cv.medianBlur(dst, 5)
    dst = cv.normalize(dst, dst=mask, alpha=minVal,
                       beta=maxVal, norm_type=cv.NORM_MINMAX)

    '''高斯滤波'''
    dst = cv.GaussianBlur(dst, (9, 9), 3)

    '''二值化处理'''
    _, threshold = cv.threshold(dst, 100, 255, cv.THRESH_BINARY)
    dst = cv.adaptiveThreshold(
        dst, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 101, -1)

    binary = dst

    '''加法去黑边'''
    dst = cv.add(dst, threshold)

    '''降噪处理·滤波操作'''

    # dst = cv.medianBlur(dst, 13)

    # dst = cv.filter2D(dst, -1, kernel=kernel)

    # dst = cv.morphologyEx(dst, cv.MORPH_OPEN, kernel=kernel)  # 开操作降噪

    '''----------------------------------霍夫直线检测-----------------------------------'''

    '''截断染色左右两边'''
    cut = dye(dst, 8, 8)
    hough(src, cut)

    '''检测左右晶线'''
    img = cv.flip(dst, 1)
    hough_img2, img2_line = hough(
        np.zeros([src.shape[0], src.shape[1], 3], src.dtype), img)

    hough_img1, img1_line = hough(
        np.zeros([src.shape[0], src.shape[1], 3], src.dtype), cut)

    '''使用镜像翻转检测右半图晶线'''
    hough_img2 = cv.flip(hough_img2, 1)

    if img2_line != None:
        img2_line = (img.shape[1] - img2_line[0], img2_line[1],
                     img.shape[1] - img2_line[2], img2_line[3])

    if not lines_crossed(img1_line, img2_line):
        hough_img = cv.add(hough_img1, hough_img2)
    else:
        hough_img = hough_img1

    houghP_img, houghP_lines = houghP(
        np.zeros([src.shape[0], src.shape[1], 3], src.dtype), dst, 40)

    houghP_result = houghP_img

    '''绘制直线与绿线交点'''
    if not lines_crossed(img1_line, img2_line):
        if img1_line != None:
            for houghP_line in houghP_lines:
                point_is_exsit, point = lines_crossed(
                    img1_line, houghP_line, 2)
                if point_is_exsit:
                    cv.circle(houghP_result, point, 1, (0, 0, 255), 2)

        if img2_line != None:
            for houghP_line in houghP_lines:
                point_is_exsit, point = lines_crossed(
                    img2_line, houghP_line, 2)
                if point_is_exsit:
                    cv.circle(houghP_result, point, 1, (0, 0, 255), 2)
    else:
        for houghP_line in houghP_lines:
            point_is_exsit, point = lines_crossed(img1_line, houghP_line, 2)
            if point_is_exsit:
                cv.circle(houghP_result, point, 1, (0, 0, 255), 2)

    hough_result = cv.add(hough_img, src)
    result = cv.add(houghP_result, hough_result)
    show("result", result)

    '''----------------------------------写入结果-----------------------------------'''

    cv.imwrite(save_path + img_path[-11:-4] + "/preview.bmp", preview)
    cv.imwrite(save_path + img_path[-11:-4] + "/binary.bmp", binary)
    cv.imwrite(save_path + img_path[-11:-4] + "/cut.bmp", cut)
    cv.imwrite(save_path + img_path[-11:-4] + "/result.bmp", result)

    return result


if __name__ == '__main__':
    img_path = "D:/study/opencv/TEST"
    save_path = "D:/study/opencv/TEST/"

    path_lst = [i for i in os.listdir(img_path) if i.endswith(".bmp")]

    # for i in path_lst:
    #     process(img_path + "/" + i)
    process(img_path + "/1444200.bmp", save_path)

    while True:
        c = cv.waitKey(50)
        if c == 27:
            break
    cv.destroyAllWindows()
