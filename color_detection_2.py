# Wrong method of approach to color detection

import cv2 as cv
import numpy as np

webcam = cv.VideoCapture(0)
unique_colors = set()
detected_colors = []

while True:
    _, imageFrame = webcam.read()
    hsvFrame = cv.cvtColor(imageFrame, cv.COLOR_BGR2HSV)

    hist = cv.calcHist([hsvFrame], [0], None, [180], [0, 180])
    dominant_hue = np.argmax(hist)
    
    lower_bound = np.array([dominant_hue - 10, 50, 50], dtype=np.uint8)
    upper_bound = np.array([dominant_hue + 10, 255, 255], dtype=np.uint8)
    
    mask = cv.inRange(hsvFrame, lower_bound, upper_bound)

    # Morphological Transform, Dilation
    kernal = np.ones((5, 5), "uint8")
    mask = cv.dilate(mask, kernal) # dilate dilates the image

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv.contourArea(contour)
        if area > 100:
            # Calculate the mean color within the contour region
            mask_roi = mask.copy()
            mask_roi[mask_roi != 255] = 0
            mean_color = cv.mean(hsvFrame, mask=mask_roi)
            hue = int(mean_color[0])

            # Determine the color label based on the dominant hue
            color_label = None
            if 0 <= hue <= 10 or 170 <= hue <= 180:
                color_label = "Red"
            elif 26 <= hue <= 35:
                color_label = "Yellow"
            elif 36 <= hue <= 85:
                color_label = "Green"
            elif 86 <= hue <= 125:
                color_label = "Blue"
            elif 126 <= hue <= 150:
                color_label = "Purple"
            elif 151 <= hue <= 175:
                color_label = "Pink"
            else:
                color_label = "Unknown"

            detected_colors.append((hue, color_label))

    print(detected_colors)

    cv.imshow('Video Feed', imageFrame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        webcam.release()
        cv.destroyAllWindows()
        break