import numpy as np
import cv2
import matplotlib.pyplot as plt

def sift_kp(image):
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    sift =  cv2.SIFT_create()
    kp,des = sift.detectAndCompute(image,None)
    kp_image = cv2.drawKeypoints(gray_image,kp,None)
    return kp_image,kp,des

def get_good_match(des1,des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good

def siftImageAlignment(img1,img2):
   _,kp1,des1 = sift_kp(img1)
   _,kp2,des2 = sift_kp(img2)
   goodMatch = get_good_match(des1,des2)
   img3 = cv2.drawMatches(img1=img1,
                       keypoints1=kp1,
                       img2=img2,
                       keypoints2=kp2,
                       matches1to2=goodMatch[:40],
                       outImg = None) #画出匹配关系
   plt.imshow(img3)
   plt.show()

   if len(goodMatch) > 4:
       ptsA= np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
       ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
       ransacReprojThreshold = 4
       H, status =cv2.findHomography(ptsA,ptsB,cv2.RANSAC,ransacReprojThreshold);
       #其中H为求得的单应性矩阵矩阵
       #status则返回一个列表来表征匹配成功的特征点。
       #ptsA,ptsB为关键点
       #cv2.RANSAC, ransacReprojThreshold这两个参数与RANSAC有关
       imgOut = cv2.warpPerspective(img2, H, (img1.shape[1],img1.shape[0]),flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
   return imgOut,H,status


def imgMatch(src_img,match_img):


    img1 = cv2.imread(src_img)
    img2 = cv2.imread(match_img)
    print(img2.shape)
    while img1.shape[0] >  1000 or img1.shape[1] >1000:
        img1 = cv2.resize(img1,None, fx=0.5,fy=0.5,interpolation = cv2.INTER_AREA)
    while img2.shape[0] >  1000 or img2.shape[1] >1000:
        img2 = cv2.resize(img2,None, fx=0.5,fy=0.5,interpolation = cv2.INTER_AREA)
       
       
    result,_,_ = siftImageAlignment(img1,img2)
    #allImg = np.concatenate((img1,img2,result),axis=1)
    cv2.namedWindow('1',cv2.WINDOW_NORMAL)
    cv2.namedWindow('2',cv2.WINDOW_NORMAL)
    cv2.namedWindow('Result',cv2.WINDOW_NORMAL)
    cv2.imshow('1',img1)
    cv2.imshow('2',img2)
    cv2.imshow('Result',result)

    cv2.waitKey(0)
    #cv2.imshow('Result',allImg)
    if cv2.waitKey(2000) & 0xff == ord('q'):
        cv2.destroyAllWindows()
        cv2.waitKey(1)

if __name__=="__main__":
    src_img="C:/Users/sai3322111/Pictures/ee.jpg"
    dst_img="C:/Users/sai3322111/Pictures/ee2.jpg"
    imgMatch(src_img,dst_img)