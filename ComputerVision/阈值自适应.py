
import cv2
from matplotlib.colors import cnames
import matplotlib.pyplot as plt

# Load our new image
path=r"C:\Users\axjing\Downloads\5532841ac59c434eaabf7da44ded00cd.jpg"
image = cv2.imread(path, 0)
image1 = cv2.imread(path)

plt.figure(figsize=(30, 30))
plt.subplot(3, 2, 1)
plt.title("gray")
plt.imshow(image,cmap="gray")
ret,thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
plt.subplot(3, 2, 2)
plt.title("Threshold Binary")
plt.imshow(thresh1,cmap="gray")
image = cv2.GaussianBlur(image, (3, 3), 0)
thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5) 
plt.subplot(3, 2, 3)
plt.title("Adaptive Mean Thresholding")
plt.imshow(thresh,cmap="gray")
_, th2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.subplot(3, 2, 4)
plt.title("Otsu's Thresholding")
plt.imshow(th2,cmap="gray")
plt.subplot(3, 2, 5)
blur = cv2.GaussianBlur(image, (5,5), 0)
_, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.title("Guassian Otsu's Thresholding")
plt.imshow(th3,cmap="gray")

plt.subplot(3, 2, 6)
# blur = cv2.GaussianBlur(image, (5,5), 0)
# _, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.title("Raw Image")
plt.imshow(image1,cmap="gray")

plt.show()