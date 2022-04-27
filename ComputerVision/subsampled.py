import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.npyio import save


def rgb2gray(img):
    #第一种 GRAY=0.3*R+0.59*G+0.11*B
    #只有当数组类型为uint8时，opencv才会认为这是图片
    img_gray = img[:,:,0] * 0.11 + img[:,:,1] * 0.59 + img[:,:,2] * 0.3
    img_gray = img_gray.astype(np.uint8)
    return img_gray

def subsampled(img_path,save_path,sampling_multiple=1):

    raw_img=cv2.imread(img_path)

    img=rgb2gray(raw_img)
    h,w=img.shape
    print(h/sampling_multiple,w/sampling_multiple)
    img_v=img[0:sampling_multiple:h,0:sampling_multiple:w]

    L=0
    R=0
    img_down = np.zeros((int(h/sampling_multiple),int(w/sampling_multiple)),)
    # img_down = np.zeros((h,w))
    # 方法一循环遍历每一个像素点，j为行，i表示列
    for j in range(0,h,sampling_multiple):
        for i in range(0,w,sampling_multiple):
            #print("L:",L,"\tR:",R)
            img_down[L,R]=img[j,i]
            R+=1
        L+=1
        R=0
    print(img_down.shape)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.subplot(1,2,2)
    plt.imshow(img_down)
    plt.show()

    plt.figure()
    plt.imshow(img_v)
    plt.show()
    #cv2.imwrite(save_path)
    return img_down

def upsampled(path):
    img=cv2.imread(path)
    for i in range(2):

        img_up=cv2.pyrUp(img)
        cv2.imshow("pyramid_demo_%s"%i,img_up)
        img=img_up
        # plt.figure()
        # plt.imshow(img_up)
        # plt.show()
    return img_up
if __name__=="__main__":
    path=r"C:/Users/sai3322111/Pictures/ee.jpg"
    # subsampled(path,path,10)

    cv2.namedWindow("input image",cv2.WINDOW_AUTOSIZE)    #创建GUI窗口,形式为自适应
    # cv2.imshow("input image",src) 
    upsampled(path)
    cv2.waitKey(0)   #等待用户操作，里面等待参数是毫秒，我们填写0，代表是永远，等待用户操作
    cv2.destroyAllWindows()  #销毁所有窗口



