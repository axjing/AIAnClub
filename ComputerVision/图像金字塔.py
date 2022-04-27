
import sys
import cv2 as cv
import matplotlib.pyplot as plt
def main(argv):
    print("""
    Zoom In-Out
    ------------------
    * [i] -> Zoom [i]n
    * [o] -> Zoom [o]ut
    * [ESC] -> Close program
    """)
    
    filename = argv[0] if len(argv) > 0 else 'child.jpg'
    print(argv[0])
    # Load the image
    image = cv.imread(cv.samples.findFile(filename))
    # Check if image is loaded fine
    # image = cv.imread('child.jpg')
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    plt.figure(figsize=(20, 20))
    plt.subplot(2, 2, 1)
    plt.title("Original")
    plt.imshow(image)
    smaller = cv.pyrDown(image)
    larger = cv.pyrUp(image)
    plt.subplot(2, 2, 2)
    plt.title("Smaller")
    plt.imshow(smaller)
    plt.subplot(2, 2, 3)
    plt.title("Larger")
    plt.imshow(larger)
    plt.show()

    return 0
if __name__ == "__main__":
    main(sys.argv[1:])

 