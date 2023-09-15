import numpy as np
import cv2 as cv

webcam = cv.VideoCapture(0)
while True:
    _, imageFrame = webcam.read()
    hsvFrame = cv.cvtColor(imageFrame, cv.COLOR_BGR2HSV)
    
    # Red color
    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    red_mask = cv.inRange(hsvFrame, red_lower, red_upper)

    # Green color
    green_lower = np.array([25, 52, 72], np.uint8)
    green_upper = np.array([102, 255, 255], np.uint8)
    green_mask = cv.inRange(hsvFrame, green_lower, green_upper)

    # Blue color
    blue_lower = np.array([94, 80, 2], np.uint8)
    blue_upper = np.array([120, 255, 255], np.uint8)
    blue_mask = cv.inRange(hsvFrame, blue_lower, blue_upper)

    # Black color
    black_lower = np.array([0, 0, 0], np.uint8)
    black_upper = np.array([180, 255, 30], np.uint8)
    black_mask = cv.inRange(hsvFrame, black_lower, black_upper)

    # Yellow color
    yellow_lower = np.array([20, 100, 100], np.uint8)
    yellow_upper = np.array([30, 255, 255], np.uint8)
    yellow_mask = cv.inRange(hsvFrame, yellow_lower, yellow_upper)

    # Morphological Transform, Dilation
    kernal = np.ones((5, 5), "uint8")

    red_mask = cv.dilate(red_mask, kernal) # dilate dilates the image
    res_red = cv.bitwise_and(imageFrame, imageFrame, mask = red_mask) # bitwise_and is a bitwise AND operation

    green_mask = cv.dilate(green_mask, kernal)
    res_green = cv.bitwise_and(imageFrame, imageFrame, mask = green_mask)

    blue_mask = cv.dilate(blue_mask, kernal)
    res_blue = cv.bitwise_and(imageFrame, imageFrame, mask = blue_mask)

    black_mask = cv.dilate(black_mask, kernal)
    res_black = cv.bitwise_and(imageFrame, imageFrame, mask = black_mask)

    yellow_mask = cv.dilate(yellow_mask, kernal)
    res_yellow = cv.bitwise_and(imageFrame, imageFrame, mask = yellow_mask)

    # Tracking the Red Color
    contours, hierarchy = cv.findContours(red_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv.boundingRect(contour)
            imageFrame = cv.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv.putText(imageFrame, "Red", (x,y), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
    
    # Tracking the Green Color
    contours, hierarchy = cv.findContours(green_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv.boundingRect(contour)
            imageFrame = cv.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(imageFrame, "Green", (x,y), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
    
    # Tracking the Blue Color
    contours, hierarchy = cv.findContours(blue_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv.boundingRect(contour)
            imageFrame = cv.rectangle(imageFrame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv.putText(imageFrame, "Blue", (x,y), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0))

    # Tracking the Black Color
    contours, hierarchy = cv.findContours(black_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv.boundingRect(contour)
            imageFrame = cv.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 0, 0), 2)
            cv.putText(imageFrame, "Black", (x,y), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))

    # Tracking the Yellow Color
    contours, hierarchy = cv.findContours(yellow_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv.boundingRect(contour)
            imageFrame = cv.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv.putText(imageFrame, "Yellow", (x,y), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255))

    cv.imshow("Multiple Color Detection in Real-TIme", imageFrame)
    if cv.waitKey(10) & 0xFF == ord('q'):
        webcam.release()
        cv.destroyAllWindows()
        break