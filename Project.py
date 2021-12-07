import cv2 as cv
import numpy as np
import os

def contrast_boost_add(image, n=3):
    dst = image
    m = dst
    for i in range(n):
        dst = cv.add(m, dst)
    return dst

def contrast_brightness(image, c, b):
    h, w = image.shape
    blank = np.zeros([h, w], image.dtype)
    dst = cv.addWeighted(image, c, blank, 1-c, b)
    return dst


def dilate_then_erode(img, times, dilate, erode):
    dst = img
    for i in range(times):
        kerneld = cv.getStructuringElement(cv.MORPH_RECT, (1, dilate))
        dst = cv.dilate(dst, kerneld)
        kernele = cv.getStructuringElement(cv.MORPH_RECT, (erode, 1))
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
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.imshow(name, image)


# 截断染色函数，将图像的两边染色，第二个参数为左边染色1/lw，第三个参数为右边染色1/rw,第四个参数为上面的染色范围 第五个参数为染色的灰度值
def dye(image, lw=5, rw=5,  sh = 3, value = 255):
    h, w = image.shape
    mask = np.ones([h,w],np.uint8)
    mask *=value
    mask[2*int(h/sh):h,int(w/lw): int(w-w/rw)] = 0
    dst = cv.add(mask, image)
    return dst


def process(path):
    os.mkdir("D:/Program Files (x86)/Project_1/PV/test/demo/"+path[-11:-5])
    global src
    src = cv.imread(path)
    # 源图片展示
    # show("input image", src)
    dst = src

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    kerneld = cv.getStructuringElement(cv.MORPH_RECT, (1, 3))
    kernele = cv.getStructuringElement(cv.MORPH_RECT, (3, 1))
    
    '''灰度化，降色彩通道为1'''
    dst = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    # mask = numpy.zeros(src.shape)  # 黑色掩膜
    mask = np.ones(src.shape)  # 白色掩膜

    '''增强对比度'''
    dst = contrast_boost_add(dst, 2)
    show("preview", dst)
    
    cv.imwrite("D:/Program Files (x86)/Project_1/PV/test/demo/"+path[-11:-5]+"/_preview.bmp", dst)
    
    # '''寻找最大值和最小值'''
    # minVal, maxVal, minIdx, maxIdx = cv.minMaxLoc(dst)
    # dst = cv.medianBlur(dst, 5)
    # dst = cv.normalize(dst, dst=mask, alpha=minVal, beta=maxVal, norm_type=cv.NORM_MINMAX)
    
    '''高斯滤波'''
    dst = cv.GaussianBlur(dst, (9, 9), 3)
    
    '''二值化处理'''
    _, threshold = cv.threshold(dst, 100, 255, cv.THRESH_BINARY)
    # show("th", threshold)

    '''RIO·提取局部并进行处理，中间包括对比度增强与二值化'''
    # dst = Region_One_process(dst, 5, 5, contrast_boost_in)
    dst = cv.adaptiveThreshold(
        dst, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 101, -1)
    
    
    cv.imwrite("D:/Program Files (x86)/Project_1/PV/test/demo/"+path[-11:-5]+"/_binary.bmp", dst)
    
    '''加法去黑边'''
    dst = cv.add(dst, threshold)

    show("processed", dst)
    
    # # dst = cv.erode(dst, ker_erode)
    # dst = cv.dilate(dst, ker_dilate)
    # dst = cv.dilate(dst, ker_dilate)
    
    # dst = cv.erode(dst, ker_erode)



    '''降噪处理·滤波操作'''
    
    # dst = cv.medianBlur(dst, 13)
    # show("zhongzhilvbo", dst)

    # show("temp1", dst)
    # show("temp2", dst)

    # dst = cv.filter2D(dst, -1, kernel=kernel)

    # dst = cv.morphologyEx(dst, cv.MORPH_OPEN, kernel=kernel)  # 开操作降噪
    # show("kaicaozuo", dst)

    show("dst", dst)
    # '''Canny检测'''
    # edges = cv.Canny(dst, 10, 1000)
    # show("edges", edges)

    # '''截断染色左右两边'''
    # dst = dye(dst, 8, 8)
    # show("dye", dst)


    def houghP(src, dst, minLineLength):
        dst = cv.medianBlur(dst, 13)
            
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
        '''截断染色左右两边'''
        dst = dye(dst, 8, 8)
        show("dye", dst)
        
        cv.imwrite("D:/Program Files (x86)/Project_1/PV/test/demo/"+path[-11:-5]+"/_dye.bmp", dst)
        
        '''膨胀腐蚀'''
        # dst = dilate_then_erode(dst, 20, 6, 3)
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
        
        show("erode", dst)
        
        '''resize函数具有抗齿距的效果'''
        dst = cv.resize(dst, dst.shape, interpolation=cv.INTER_CUBIC)
        dst = cv.resize(dst, dst.shape, interpolation=cv.INTER_CUBIC)

        # show("temp", dst)
        
        dst = np.array(dst, dtype= np.uint8)
        '''霍夫直线检测'''
        # 形态学闭操作
        res = cv.Canny(dst, 10, 1000)
        show("canny", res)
        lines = cv.HoughLines(res, rho = 2, theta = np.pi/180, threshold = 85)
        # lines = list(filter(lambda x: 2.88 < x[0][1] < 3.4, lines))
        # 筛选并排序
        lines = sorted([line for line in lines if 2.88 < line[0][1] < 3.4], key = lambda x :x[0][1])
        # print(lines)
        background = src.copy()

        if len(lines):
            # 由极坐标系转换成的霍夫空间下的横纵坐标
            rho = lines[len(lines)//2][0][0]  # 第一个元素是距离rho
            theta = lines[len(lines)//2][0][1]  # 第二个元素是角度theta
            # 该直线与第一行的交点
            pt1 = (int(rho / np.cos(theta)), int(0))
            # 该直线与最后一行的焦点
            pt2 = (int(
                (rho - background.shape[0] * np.sin(theta)) / np.cos(theta)), int(background.shape[0]))
            cv.line(background, pt1, pt2, (255, 255, 0), 5)

        return background

    houghP_img = houghP(
        np.zeros([src.shape[0], src.shape[1], 3], src.dtype), dst, 50)
    
    # 左右检测线
    img = cv.flip(dst, 1)
    hough_img2 = hough(
        np.zeros([src.shape[0], src.shape[1], 3], src.dtype), img)
    
    hough_img1 = hough(
        np.zeros([src.shape[0], src.shape[1], 3], src.dtype), dst)
    
    hough_img2 = cv.flip(hough_img2, 1)
    
    hough_img = cv.add(hough_img1, hough_img2)

    result = cv.add(hough_img, src)
    show("result", result)
    cv.imwrite("D:/Program Files (x86)/Project_1/PV/test/demo/"+path[-11:-5]+"/_result.bmp", result)
    
    return cv.add(hough_img, houghP_img)


path_wsj = "D:/Program Files (x86)/Project_1/PV/test"
path_lst = []
for file in os.listdir(path_wsj):
    if file.endswith(".bmp"):
        path_lst.append(file)
print(path_lst)
path_lst2 = ["/1460800.bmp"]

path_zfz = "D:/study/opencv/detection/1460000.bmp"

# show("img", process(path_wsj+"/1460600.bmp"))

for i in path_lst:
    process(path_wsj+"/"+i)

while True:
    c = cv.waitKey(50)
    if c == 27:
        break
cv.destroyAllWindows()
