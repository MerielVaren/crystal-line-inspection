import numpy
import cv2


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

def contrast_boost_im(image, n = 3):  # 参数n 为自加的次数
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

'''通过cv2.addWeighted函数，做出的对比度增强'''
def contrast_brightness_demo(image, c, b):
    h, w, ch = image.shape
    # blank是一个与传入的图像参数大小相同的空白板
    blank = numpy.zeros([h, w, ch], image.dtype)
    dst = cv2.addWeighted(image, c, blank, 1-c, b)
    return dst


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
    print(shape)
    part = image[: , :]
    '''这里进行对局部的操作'''
    image[: , :] = part
    return image

'''RIO·提取局部并进行 im型的自加来增强对比度，参数为：第一个为整张图片，第二个为局部高的像素，第三个为局部行的范围'''
def Region_add_contrast_boost(image, m = 50, n = 50):
    height, width, channels = image.shape
    # 划分为 rows 行， cols列，每一行高为 m，每一列宽为 n
    rows = int(height/m)
    cols = int(width/n)
    
    '''这里进行对局部的操作'''
    for row in range(rows-1): 
        for col in range(cols-1):
            # 将局部进行对比度加强
            image[row*m : (row+1)*m ,  col*n: (col+1)*n,] = contrast_boost_im(image[row*m : (row+1)*m ,  col*n: (col+1)*n,])
    image[rows*m: height, cols*n: width,] = contrast_boost_im(image[rows*m: height, cols*n: width,])
    
    return image

'''泛洪填充'''
def fill_color_demo(image):
    copying = image.copy()
    h, w = image[:2]
    mask = numpy.zeros([h+2, w+2], numpy.uint8)
    cv2.floodFill(copying,mask,(30,30), (0, 255, 255), (100, 100, 100), (50, 50, 50), cv2.FLOODFILL_FIXED_RANGE)
    return copying


'''根据传参的路径来处理照片'''
def process(path):
    #将路径的照片文件转化成多维数组
    src = cv2.imread(path)
    # 创建窗口
    cv2.namedWindow("input image", cv2.WINDOW_AUTOSIZE)
    # 以多维数组为基础，重新构成图片并在对应的窗口展示
    cv2.imshow("input image", src)
    dst = src
    t1 =cv2.getTickCount()

    '''call your function'''

    t2 =cv2.getTickCount()
    
    cv2.imshow("result", dst)
    
    time = (t2-t1)/cv2.getTickFrequency()
    print("Time consumption: %s  ms"%(time*1000))
    # 延迟时间
    cv2.waitKey(0)
    # 将数组以图片形式保存在指定目录下
    cv2.imwrite(path+"/result.png", dst)
    # 关闭所有窗口
    cv2.destroyAllWindows()
    print("end")


