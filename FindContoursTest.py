import cv2
import numpy as np

MIN_CONTOUR_AREA = 100

train_image = cv2.imread("HI.png")

gray_image = cv2.cvtColor(train_image, cv2.COLOR_BGR2GRAY)
blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

bw_image = cv2.adaptiveThreshold(blur_image, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV,
                                 11, 2)

bw_image_copy = bw_image.copy()

npa_contours, npa_hierarchy = cv2.findContours(bw_image_copy,
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)

for npa_contour in npa_contours:
    if cv2.contourArea(npa_contour) > MIN_CONTOUR_AREA:  # if contour is big enough to consider
        [intX, intY, intW, intH] = cv2.boundingRect(npa_contour)
        cv2.rectangle(train_image,  # draw rectangle on original training image
                      (intX, intY),  # up left
                      (intX + intW, intY + intH),  # down right
                      (0, 0, 255),  # red
                      2)  # thickness
valid_contours = []
for i in range(0, np.size(npa_contours)):
    if cv2.contourArea(npa_contours[i]) > MIN_CONTOUR_AREA:
        valid_contours.append(npa_contours[i])
print(np.size(valid_contours))

#for i in range(0, np.size(npa_contours)):
# cv2.drawContours(train_image, npa_contours, -1, (0, 255, 0), 3)
cv2.imshow("training_numbers.png", train_image)
cv2.waitKey(0)

print("\n\ntraining complete !!\n")
