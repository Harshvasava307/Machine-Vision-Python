import cv2
import numpy as np

def cartoon(img_path):
    # Read the image using OpenCV
    img = cv2.imread(img_path)
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply the cartoon effect using the cv2.stylization function
    gray = cv2.medainBlur(gray,5)
    
    color = cv2.bilateralFilter(img,9,20,250)
    cartoon = cv2.bitwise_and(color, color, mask=gray)
    
    cv2.imshow("Cartoon",cartoon)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

cartoon("test.jpg")
