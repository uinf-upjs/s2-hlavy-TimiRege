import os

import pathlib
import matplotlib.pyplot as plt
import cv2
import numpy as np

input_dir = pathlib.Path(r'C:\Users\hp\PycharmProjects\ANO\exam\input')
pics = os.listdir(input_dir)
for item in pics:
    img = cv2.imread(f'C:/Users/hp/PycharmProjects/ANO/exam/input/{item}', cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.medianBlur(gray, 9)
    cv2.imwrite(f'C:/Users/hp/PycharmProjects/ANO/exam/output/{item}_blur.jpg', blurred)

    equalized_img = cv2.equalizeHist(gray)
    cv2.imwrite(f'C:/Users/hp/PycharmProjects/ANO/exam/output/{item}_eq.jpg', equalized_img)

    canny = cv2.Canny(equalized_img, 20, 55)
    cv2.imwrite(f'C:/Users/hp/PycharmProjects/ANO/exam/output/{item}_canny.jpg', canny)

    rows = gray.shape[0]
    circles = cv2.HoughCircles(equalized_img, cv2.HOUGH_GRADIENT, 1, rows / 8,
                               param1=55, param2=20,
                               minRadius=10, maxRadius=25)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(img, center, 1, (0, 100, 100), 2)
            # circle outline
            radius = i[2]
            cv2.circle(img, center, radius, (255, 255, 0), 2)

    cv2.imwrite(f'C:/Users/hp/PycharmProjects/ANO/exam/output/{item}_circ.jpg', img)










