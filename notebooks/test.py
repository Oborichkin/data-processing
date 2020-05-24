# import the necessary packages
import cv2

image = cv2.imread("../data/stones.jpg")
shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
cv2.imshow("Input", image)
