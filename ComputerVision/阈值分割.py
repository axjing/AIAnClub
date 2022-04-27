import cv2
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil

def Threshold(im):
    image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # 将图像转为灰色
    blurred = cv2.GaussianBlur(image, (5, 5), 0)  # 高斯滤波
    # cv2.imshow("Image", image)  # 显示图像
    (T, thresh) = cv2.threshold(blurred, 155, 255, cv2.THRESH_BINARY)  # 阈值化处理，阈值为：155
    # cv2.imshow("Threshold Binary", thresh)

    (T, threshInv) = cv2.threshold(blurred, 155, 255, cv2.THRESH_BINARY_INV)  # 反阈值化处理，阈值为：155
    # cv2.imshow("Threshold Binary Inverse", threshInv)

    # cv2.imshow("Coins", cv2.bitwise_and(image, image, mask =threshInv))
    # cv2.waitKey(0)

    # 阈值化处理
    ret, thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(image, 127, 255, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO_INV)

    # 显示结果
    titles = ['Gray Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    images = [image, thresh1, thresh2, thresh3, thresh4, thresh5]
    for i in range(6):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == '__main__':

    img_path = r"C:/Users/axjin/Downloads/1-1.jpeg"
    im=cv2.imread(img_path)
    Threshold(im)
    print("*" * 30 + "\n |\t\tEnd Of Program\t\t|\n" + "*" * 30)