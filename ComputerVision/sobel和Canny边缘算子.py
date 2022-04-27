import cv2
import matplotlib.pyplot as plt


if __name__=="__main__":

    image = cv2.imread('shuiguo.jpeg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hgt, wdt,_ = image.shape
    # Sobel Edges
    x_sobel = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    y_sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    plt.figure(figsize=(20, 20))
    plt.subplot(3, 2, 1)
    plt.title("Original")
    plt.imshow(image)
    plt.subplot(3, 2, 2)
    plt.title("Sobel X")
    plt.imshow(x_sobel)
    plt.subplot(3, 2, 3)
    plt.title("Sobel Y")
    plt.imshow(y_sobel)
    sobel_or = cv2.bitwise_or(x_sobel, y_sobel)
    plt.subplot(3, 2, 4)
    plt.imshow(sobel_or)
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    plt.subplot(3, 2, 5)
    plt.title("Laplacian")
    plt.imshow(laplacian)
    ## There are two values: threshold1 and threshold2.
    ## Those gradients that are greater than threshold2 => considered as an edge
    ## Those gradients that are below threshold1 => considered not to be an edge.
    ## Those gradients Values that are in between threshold1 and threshold2 => either classiﬁed as edges or non-edges
    # The first threshold gradient
    canny = cv2.Canny(image, 50, 120)
    plt.subplot(3, 2, 6)
    plt.imshow(canny)
    plt.show()