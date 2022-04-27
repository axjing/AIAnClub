'''
形态学操作
简言之：一组基于形状处理图像的操作。形态操作将结构元素应用于输入图像并生成输出图像。
最基本的形态操作是：侵蚀和膨胀。它们具有广泛的用途，即：
消除噪音
隔离单个元素并连接图像中的不同元素。
在图像中查找强度凸起或孔
我们将用以下图像简要解释扩张和侵蚀：
'''

from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse

# 形态学算子的形状映射
def morph_shape(val):
    if val == 0:
        return cv.MORPH_RECT
    elif val == 1:
        return cv.MORPH_CROSS
    elif val == 2:
        return cv.MORPH_ELLIPSE

# 腐蚀算子
def erosion(val):
    erosion_size = cv.getTrackbarPos(title_trackbar_kernel_size, title_erosion_window)
    erosion_shape = morph_shape(cv.getTrackbarPos(title_trackbar_element_shape, title_erosion_window))
    
    element = cv.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                       (erosion_size, erosion_size))
    
    erosion_dst = cv.erode(src, element)
    cv.imshow(title_erosion_window, erosion_dst)

# 膨胀算子
def dilatation(val):
    dilatation_size = cv.getTrackbarPos(title_trackbar_kernel_size, title_dilation_window)
    dilation_shape = morph_shape(cv.getTrackbarPos(title_trackbar_element_shape, title_dilation_window))
    element = cv.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                       (dilatation_size, dilatation_size))
    dilatation_dst = cv.dilate(src, element)
    cv.imshow(title_dilation_window, dilatation_dst)



src = None
erosion_size = 0
max_elem = 2
max_kernel_size = 21
title_trackbar_element_shape = '形状映射:\n 0: Rect \n 1: Cross \n 2: Ellipse'
title_trackbar_kernel_size = 'Kernel size:\n 2n +1'
title_erosion_window = 'Erosion'
title_dilation_window = 'Dilation'
def main(image):
    global src
    src = cv.imread(cv.samples.findFile(image))
    if src is None:
        print('Could not open or find the image: ', image)
        exit(0)
    cv.namedWindow(title_erosion_window)
    cv.createTrackbar(title_trackbar_element_shape, title_erosion_window, 0, max_elem, erosion)
    cv.createTrackbar(title_trackbar_kernel_size, title_erosion_window, 0, max_kernel_size, erosion)
    cv.namedWindow(title_dilation_window)
    cv.createTrackbar(title_trackbar_element_shape, title_dilation_window, 0, max_elem, dilatation)
    cv.createTrackbar(title_trackbar_kernel_size, title_dilation_window, 0, max_kernel_size, dilatation)
    erosion(0)
    dilatation(0)
    cv.waitKey()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='膨胀、腐蚀案例.')
    parser.add_argument('--input', help='Path to input image.', default='./pepole.jpeg')
    args = parser.parse_args()
    main(args.input)