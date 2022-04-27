import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


if __name__=="__main__":

    img = cv.imread('notebook.jpg')
    rows,cols,ch = img.shape
    print(rows,cols,ch)
    pts1 = np.float32([[624,503],[2541,237],[318,3351],[2620,3630]])
    pts2 = np.float32([[0,0],[3000,0],[0,4000],[3000,4000]])
    M = cv.getPerspectiveTransform(pts1,pts2)
    dst = cv.warpPerspective(img,M,(3000,4000))
    plt.subplot(121),plt.imshow(img),plt.title('Input')
    plt.subplot(122),plt.imshow(dst),plt.title('Output')
    plt.show()