import cv2
import numpy as np
import sys

def onTrackbarChange(max_slider):
 
    path="G:/gonglu.png"
    img = cv2.imread(path, 1)
    flag=0

    # Convert to gray-scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    p1 = max_slider
    p2 = max_slider * 0.4
    # Edge image for debugging
    edges = cv2.Canny(img, p1, p2)
    if flag==0:
        # Detect circles using HoughCircles transform
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, img.shape[0]/64, param1=p1, param2=p2, minRadius=25, maxRadius=50)
        # If at least 1 circle is detected
        if circles is not None:
            cir_len = circles.shape[1] # store length of circles found
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # Draw the outer circle
                cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # Draw the center of the circle
                cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
        else:
            cir_len = 0 # no circles detected
    elif flag==1:
        # Apply probabilistic hough line transform
        lines = cv2.HoughLinesP(edges, 2, np.pi/180.0, 50, minLineLength=10, maxLineGap=100)
        # Draw lines on the detected points
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0,0,255), 1)

    
    # Display output image
    cv2.imshow('Image', img)    
    cv2.imshow('Edges', edges)

    

    
if __name__ == "__main__":
    # Read image
    # img = cv2.imread(sys.argv[1], 1)
    

    # Create display windows
    cv2.namedWindow("Edges")
    cv2.namedWindow("Image")
    

    # Trackbar will be used for changing threshold for edge 
    initThresh = 500 
    maxThresh = 1000 

    # Create trackbar
    cv2.createTrackbar("Threshold", "Image", initThresh, maxThresh, onTrackbarChange)
    onTrackbarChange(initThresh)

    
    while True:
        key = cv2.waitKey(1)
        if key == 27:
            break

    cv2.destroyAllWindows()