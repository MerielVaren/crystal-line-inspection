import numpy
import cv2
import math


'''打印照片信息，包括高度，宽度，色彩通道'''
def get_image_info(image):  # 传入一个numpy.ndarray类型参数,也可能传入不是的ndarray，而是其他多维数组
    print(type(image))
    print(image.shape)  # 输出高度， 宽度， 色彩通道
    print(image.size)  # 像素数据，像素个数*每个像素蕴含的数据数
    print(image.dtype)  # image是个多维数组，dtype是这个数组的最里层元素的类型(即 uint8)
    #   **ndarray**对于图片的存储形式：	
    # ​	第一层元素个数为高度，即行的个数
    # ​	第二层元素个数为宽度，即那一行像素点的个数
    # ​	宽：[ [像素点数据],[像素点数据]···················]	
    # ​	高：[ [宽],[宽],[宽],[宽]···································]
    pixel_data = numpy.array(image)  # 以image为模板创建一个数组，类型为numpy.ndarray
    return pixel_data


'''开启电脑的摄像头捕捉画面'''
def video_demo():  # 开启电脑摄像头并进行捕捉(虽说我的电脑没摄像头)
    capture = cv2.VideoCapture(0)
    while(True):
        ret, frame = capture.read()
        cv2.flip(frame, 1)  # 参数2： 1是左右调换，-1是上下颠倒
        cv2.imshow("video", frame)
        c = cv2.waitKey(5)  # 传入的参数控制摄像头捕捉的间隔
        if 27 == c:
            break

'''自动展示'''
def nothing():
    pass
def show(name, image):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)

'''根据传参的路径捕捉视频并加以处理，二值化'''
def extrace_object_demo(Vpath):
	#读取Vpath路径的视频
    capture = cv2.VideoCapture(Vpath)
    while(True):
        # ret 为bool值，读取帧数正确则返回True,读取到视频尾部则返回 False
        # frame为 ndarray类型，一个三维数组
        ret, frame = capture.read()
        if ret == False:
            break
         # 色彩空间转换
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
         # 进行处理的操作， 这里是将HSV中偏绿色的部分凸显出来，变成255，其余为0
        lower_hsv = numpy.array([37, 43, 46])
        upper_hsv = numpy.array([77, 255, 255])
        mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)
        # 原视频展示
        cv2.imshow("video", frame)
        # 处理后的视频展示
        cv2.imshow("mask", mask)
        c = cv2.waitKey(40)
        if c == 27:
            break


'''传入图片，分离色彩通道, 分别操作'''
def channel_split(image):
    b, g, r = cv2.split(image)
    # b,g,r中的每一个像素数据从数组变成了 单个色彩空间的强度值
    # cv2.imshow("blue", b)
    '''在此对不同的色彩通道进行操作'''
    intergration = cv2.merge([b, g, r])
    cv2.imshow("intergrated", intergration)


'''ndarray数组（图片）初始化'''
def ndarrayInitiation():
    # 初始化中，dtype类型默认是float64, 色彩通道默认是一个，
    m1 = numpy.ones([3,3], numpy.uint8)
    m1.fill(123)
    print(m1)
    # 这里生成的ndarrar,等价于色彩空间中仅保留绿色通道
    green = numpy.ones([400, 400, 3], numpy.uint8)
    green[:, :, 0] = numpy.ones([400, 400])*255
    m2 = numpy.zeros([400, 400, 3], numpy.uint8)
    print(m2)
    m3 = numpy.ones([400, 400])*123
    print(m3)
    m4 = m1.reshape([1, 9])
    print(m4)


'''图像的inverse处理,bit_wise_not操作'''
def BitWiseNot(image):
    # height = image.shape[0]
    # width = image.shape[1]
    # channels = image.shape[2]
    # for row in range(height):
    #     for col in range(width):
    #         for c in range(channels):
    #             pv = image[row, col, c]
    #             image[row, col, c] = 255-pv
    # 或者直接调用 已有的函数
    image = cv2.bitwise_not(image)
    cv2.imshow("BitWiseNot", image)


'''通过cv2.add函数，做出的对比度增强'''
def contrast_boost(image, n = 3):
    M = numpy.uint8(cv2.mean(image))
    dst = image
    m = dst
    for i in range(n):
        dst = cv2.add(m, dst)
    return dst

def contrast_boost_add(image, n=3):
    dst = image
    m = dst
    for i in range(n):
        dst = cv2.add(m, dst)
    return dst
    
def contrast_boost_im(image, n = 3):  # 参数n 为自加的次数, 三色彩通道
    M = numpy.uint8(cv2.mean(image))
    #三个色彩通道的均值
    dst = image
    #减去均值（即减去放大电路中的Vbb，）
    dst[:,:,0] -= M[0]
    dst[:,:,1] -= M[1]
    dst[:,:,2] -= M[2]
    m = dst
    #自加操作（类比于放大电路中的交流信号的放大）
    for i in range(n):
        dst = cv2.add(m, dst)
    #再加上均值（相当于放大电路中的输出电压）
    dst[:,:,0] += M[0]
    dst[:,:,1] += M[1]
    dst[:,:,2] += M[2]
    return dst

def contrast_boost_in(image, n = 3):  # 灰度化后的单色彩通道
    M = numpy.uint8(cv2.mean(image))[0]
    #色彩通道的均值
    dst = image
    #减去均值（即减去放大电路中的Vbb，）
    dst -= M
    m = dst
    #自加操作（类比于放大电路中的交流信号的放大）
    for i in range(n):
        dst = cv2.add(m, dst)
    #再加上均值（相当于放大电路中的输出电压）
    dst += M
    # 二值化处理,因为我们会加上均值，所以这里的二值化的下界是填的(M+5)
    ret, dst = cv2.threshold(dst, M, 255, cv2.THRESH_BINARY)
    return dst

def contrast_boost_ib(image, n = 3):
    M = numpy.int8(cv2.mean(image))[0]
    #色彩通道的均值
    dst = numpy.int8(image)
    #减去均值（即减去放大电路中的Vbb，）
    dst -= M
    m = dst
    #自加操作（类比于放大电路中的交流信号的放大）
    for i in range(n):
        dst = cv2.add(m, dst)
    #再加上均值（相当于放大电路中的输出电压）
    dst += M
    # 二值化处理,因为我们会加上均值，所以这里的二值化的下界是填的(M+5)
    ret, dst = cv2.threshold(dst, M, 255, cv2.THRESH_BINARY)
    return dst

def contrast_boost_iv(image, c = 3):
     #灰度图专属
    h,w = image.shape
    new_img = numpy.zeros((h, w))
    for i in range(h):
        for j in range(w):
            new_img[i, j] = c * (math.log(1.0 + image[i, j]))


    new_img = cv2.normalize(new_img,new_img,0,255,cv2.NORM_MINMAX)

    return new_img

'''通过cv2.addWeighted函数，做出的对比度增强'''
def contrast_brightness_demo(image, c, b):
    h, w, ch = image.shape
    # blank是一个与传入的图像参数大小相同的空白板
    # c相当于一个权重，权重大的图像更实，权重小的图像更透明
    blank = numpy.zeros([h, w, ch], image.dtype)
    dst = cv2.addWeighted(image, c, blank, 1-c, b)
    return dst


'''二值化操作'''
def threshold_demo(image):
    ret, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary
def local_threshold(image):  # 自适应阈值
    # 第三个参数有：ADAPTIVE_THRESH_GAUSSIAN_C，ADAPTIVE_THRESH_MEAN_C
    binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 23, -1)
    return binary

'''色彩空间互相转换调试'''     
def color_space_demo(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", gray)
    '''RGB到HSV的色彩空间转换很重要,其中H通道是 0-180'''
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imshow("hsv", hsv)
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    cv2.imshow("yuv", yuv)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    cv2.imshow("ycrcb", ycrcb)

'''一堆算术运算'''
def add(image1, image2):
    dst = cv2.add(image1, image2)
    return dst
def sub(image1, image2):
    dst = cv2.subtract(image1, image2)
    return dst
def divide(image1, image2):
    dst = cv2.divide(image1, image2)
    return dst
def mutiply(image1, image2):
    dst = cv2.multiply(image1, image2)
    return dst
def Mean_RMS(image):
    # 获得平均值和方差，RMS，方差是衡量对比度的重要标准
    M = cv2.mean(image)
    m, RMS = cv2.meanStdDev(image)


'''一堆逻辑运算'''
def logic_add(image1, image2):
    dst = cv2.bitwise_or(image1, image2)
    return dst
def logic_mutiply(image1, image2):
    dst = cv2.bitwise_and(image1, image2)
    return dst
def logic_not(image1, image2):
    dst = cv2.bitwise_not(image1, image2)
    return dst
def logic_xor(image1, image2):
    dst = cv2.bitwise_xor(image1, image2)
    return dst


'''RIO·提取局部图片并进行操作'''
def Region_demo(image):
    shape = image.shape
    # print(shape)
    part = image[: , :]
    '''这里进行对局部的操作'''
    image[: , :] = part
    return image

'''三通道下 RIO·提取局部并进行处理，参数为：第一个为整张图片，第二个为局部高的像素，第三个为局部行的范围，第四个参数为处理函数选择'''
def Region_Three_process(image, m = 50, n = 50, func = contrast_boost_im):
    height, width  = image.shape
    # 划分为 rows 行， cols列，每一行高为 m，每一列宽为 n
    rows = int(height/m)
    cols = int(width/n)
    
    '''这里进行对局部的操作'''
    for row in range(rows-1): 
        for col in range(cols-1):
            # 将局部进行操作
            image[row*m : (row+1)*m ,  col*n: (col+1)*n,] = \
                func(image[row*m : (row+1)*m ,  col*n: (col+1)*n,])
        # 防止不是整数，将图像边框处也处理了
        image[row*m: (row+1)*m, cols*n: width,] = \
            func(image[row*m: (row+1)*m, cols*n: width,])
    for col in range(cols-1):
        image[rows*m:height, col*n: (col+1)*n,] = \
            func(image[rows*m:height, col*n: (col+1)*n,])
    image[rows*m:height, cols*n: width,] = \
        func(image[rows*m:height, cols*n: width,])
    
    return image

'''单通道下 RIO·提取局部并进行处理，参数为：第一个为整张图片，第二个为局部高的像素，第三个为局部行的范围，第四个参数为处理函数选择'''
def Region_One_process(image, m = 50, n = 50, func = contrast_boost_in):
    height, width  = image.shape
    # 划分为 rows 行， cols列，每一分块图像的像素高为m,宽为n
    rows = int(height/m)
    cols = int(width/n)
    
    '''这里进行对局部的操作'''
    for row in range(rows-1): 
        for col in range(cols-1):
            # 将局部进行操作
            # 第row行，第col列 的 图像分块
            image[row*m:(row+1)*m ,  col*n:(col+1)*n] = \
                func(image[row*m:(row+1)*m, col*n: (col+1)*n])
        # 处理图像的右侧边框
            image[row*m:(row+1)*m, cols*n:width] = \
                func(image[row*m:(row+1)*m, cols*n:width])
    # 处理图像的底部边框
    for col in range(cols-1):
        image[rows*m:height, col*n:(col+1)*n] = \
            func(image[rows*m:height, col*n: (col+1)*n])
    # 处理图像的右下角
    image[rows*m:height, cols*n: width] = \
        func(image[rows*m:height, cols*n: width])
        
    return image


'''泛洪填充·两个不同的flag模式'''
def fill_color_demo(image):  # 填充全部
    copying = image.copy()
    h, w = image[:2]
    mask = numpy.zeros([h+2, w+2], numpy.uint8)
    cv2.floodFill(copying,mask,(30,30), (0, 255, 255), (100, 100, 100), (50, 50, 50), cv2.FLOODFILL_FIXED_RANGE)
    return copying
def fill_binary(image):  # 填充掩面中 值为 0 的部分
    h, w = image.shape[:2]
    # 掩面，填充颜面中 值为0的部分
    mask = numpy.ones([h+2, w+2, 1], numpy.uint8)
    mask[int(h/4):int(h/2), int(w/4):int(w/2)] = 0
    
    cv2.floodFill(image, mask, (int(h/3),int(w/3)), (100, 255, 255), cv2.FLOODFILL_MASK_ONLY)
    return image


'''模糊系列去噪操作'''
def blur_demo(image):  # 横向与纵向的模糊
    # 第二个参数中的第一个为横向的模糊程度
    # 第二个参数中的第二个为纵向的模糊程度
    image = cv2.blur(image, (1, 35))  
    return image
def median_blur(image):  # 去除椒盐噪声
    dst = cv2.medianBlur(image, 5)
    return dst
# 对 2D 图像实施低通滤波，帮助我们去除噪音，模糊图像
def sharpen(image):
    # cv2.filter2D(src,dst,depth,kernel,auchor=(-1,-1))：
    # dst:目标图像，与原图像尺寸和通过数相同
    # depth:目标图像的所需深度
    # kernel是卷积核，函数通过这个对整体图像进行一个单独拆分分析后叠加过程，作用是消除相邻像素值之间的差异性
    # 将核放在图像的一个像素 A 上，求与核对应的图像上 25（5x5）
    # 个像素的和，在取平均数，用这个平均数替代像素 A 的值
    # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    # kernel = np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]], np.float32)
    # kernel = np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]], np.float32)
    # kernel = np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]], np.float32)
    kernel = numpy.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]],
                      numpy.float32)
    dst = cv2.filter2D(image, -1, kernel=kernel)
    return dst

def clamp(pv):
    if pv > 255:
        return 255
    if pv < 0:
        return 0
    return pv
def gaussian_noise(image):  # 模糊图像，减少噪声的影响
    h, w, c = image.shape
    for row in range(h):
        for col in range(w):
            s = numpy.random.normal(0, 20, 3)
            b = image[row, col, 0]  # blue
            g = image[row, col, 1]  # green
            r = image[row, col, 2]  # red
            image[row, col, 0] = clamp(b+s[0])
            image[row, col, 1] = clamp(g+s[0])
            image[row, col, 2] = clamp(r+s[0])
    return image
def gaussian_demo(image):  # API 高斯模糊
    # cv2.GaussianBlur(src, ksize, sigmaX, dst=None, sigmaY=None, borderType=None)
    # ksize:高斯内核，也是卷积核；
    # （此卷积核是看权重的（中间元素的权重最大），此权重仅仅考虑了像素的空间分布）
    # sigmaX: X方向的高斯核标准差；
    # sigmaY: Y方向的高斯核标准差；
    # 这个标准差决定了图像有多糊
    # cv2.bilateralFilter()
    dst = cv2.GaussianBlur(image, (5, 5), 3)
    return dst


# 边缘保留滤波EPF： edge,preserve,filter
def bi_demo(image):  # 高斯双边
    # 高斯双边滤波：（这种操作与其他滤波器相比会比较慢）
    # 同时使用空间高斯权重和灰度值相似性高斯权重
    # 原理：在考虑像素空间上的权重问题，
    # 同时考虑了相邻像素的强度值之间的差异
    # 而差异很大往往是在图像的边缘部分。
    dst = cv2.bilateralFilter(image, 0, 100, 15)
    return dst
def shift_demo(image):  # 均值迁移(3通道)
    # 有种油画的效果
    dst = cv2.pyrMeanShiftFiltering(image, 10, 50)
    return dst


'''膨胀与腐蚀'''
def erode_demo(image):  # 腐蚀
    ret, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    dst = cv2.erode(binary, kernel=kernel)
    return dst
def dilate_demo(image):  # 膨胀
    ret, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    dst = cv2.dilate(binary, kernel=kernel)
    return dst


'''track_use'''
def track_bar(name, image):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.createTrackbar('c', name, 10, 100, nothing)
    dst = image
    while(True):
        cv2.imshow(name, dst)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        c = cv2.getTrackbarPos("c", name)
        dst = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 2*c-1, -1)


'''霍夫直线检测'''
def Hough(image):  # 对二值图像进行直线检测
    # 将图像中较为困难的全局检测问题转换为参数空间中相对容易解决的局部峰值检测问题
    minLineLength = 15
    edges = cv2.Canny(image, 10, 1000)
    cv2.imshow("edges", edges)
    hough = image.copy()
    lines = cv2.HoughLinesP(
        edges, 0.6, numpy.pi / 180, threshold=minLineLength, minLineLength=minLineLength, maxLineGap=6)
    for x1, y1, x2, y2 in lines[:, 0]:
        cv2.line(hough, (x1, y1), (x2, y2), (0, 255, 0), 2)








'''这个是主调函数·根据传参的路径来处理照片'''
def process(path):
    #将路径的照片文件转化成多维数组
    src = cv2.imread(path)
    # 源图片展示
    show("input image", src)
    dst = src
    # 计时函数，开始计时
    t1 =cv2.getTickCount()
    
    '''call your functions'''

    # show("temp", dst)
    
    '''灰度化，降色彩通道为1'''
    dst = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # mask = numpy.zeros(src.shape)  # 黑色掩膜
    mask = numpy.ones(src.shape)  # 白色掩膜

    '''增强对比度'''
    dst = contrast_boost_add(dst, 2)          
    # dst = cv2.medianBlur(dst, 5)
    dst = cv2.GaussianBlur(dst, (5, 5), 3)
    
    
    '''RIO·提取局部并进行处理，中间包括对比度增强与二值化'''
    # dst = Region_One_process(dst, 5, 5, contrast_boost_in)
    dst = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, -1)
    
    show("first", dst)
    
    '''降噪处理·滤波操作'''
    dst = cv2.medianBlur(dst, 5)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel=kernel)  # 开操作降噪
    dst = cv2.filter2D(dst, -1, kernel=kernel)
    dst = cv2.normalize(dst, dst=mask, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    dst = cv2.GaussianBlur(dst, (5, 5), 3)

    show("second", dst)
    
    '''膨胀与腐蚀 与 中途去噪''' 
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT,(4, 1))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 4))
    dst = cv2.erode(dst, kernel=kernel_erode)
    dst = cv2.dilate(dst, kernel=kernel_dilate)
    dst = cv2.erode(dst, kernel=kernel_erode)
    dst = cv2.dilate(dst, kernel=kernel_dilate)
    dst = cv2.GaussianBlur(dst, (5, 5), 3)
    
    dst = cv2.erode(dst, kernel=kernel_erode)
    dst = cv2.dilate(dst, kernel=kernel_dilate)
    dst = cv2.erode(dst, kernel=kernel_erode)
    dst = cv2.dilate(dst, kernel=kernel_dilate)
    
    dst = cv2.GaussianBlur(dst, (5, 5), 3)
    
    dst = cv2.dilate(dst, kernel=kernel_dilate)
    dst = cv2.dilate(dst, kernel=kernel_dilate)
    dst = cv2.dilate(dst, kernel=kernel_dilate)
    dst = cv2.dilate(dst, kernel=kernel_dilate)
    dst = cv2.dilate(dst, kernel=kernel_dilate)
    
    dst = cv2.normalize(dst, dst=mask, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    dst = cv2.medianBlur(dst, 5)
    

    t2 =cv2.getTickCount()
    
    show("result", dst)
    
    time = (t2-t1)/cv2.getTickFrequency()
    print("Time consumption: %s  ms"%(time*1000))
    # 延迟时间
    cv2.waitKey(0)
    # 将数组以图片形式保存在指定目录下
    cv2.imwrite( "D:/Program Files (x86)/Project_1/PV" + "/result.png", dst)
    # 关闭所有窗口
    cv2.destroyAllWindows()
    print("end")


path = "D:/Program Files (x86)/Project_1/PV/test"
sub_path1 = "/1425800.bmp"
sub_path2 = "/1668600.jpg"
sub_path3 = "/1669200.jpg"
sub_path4 = "/1462000.bmp"

process(path + sub_path4)

