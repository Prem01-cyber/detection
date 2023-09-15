import cv2 as cv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

webcam = cv.VideoCapture(0)
k_neighbors = 3

colors_to_detect = {
    "Red": ([0, 100, 100], [10, 255, 255]),
    "Green": ([40, 40, 40], [80, 255, 255]),
    "Blue": ([90, 50, 50], [130, 255, 255]),
    "Yellow": ([20, 100, 100], [30, 255, 255]),
    "Black": ([0, 0, 0], [180, 255, 30]),
    "White": ([0, 0, 200], [180, 30, 255]),
}

color_labels = list(colors_to_detect.keys())
knn_classifier = KNeighborsClassifier(n_neighbors=k_neighbors)

training_data = []
training_labels = []

for color, (lower, upper) in colors_to_detect.items():
    lower = np.array(lower, dtype=np.uint8)
    upper = np.array(upper, dtype=np.uint8)

    # Convert HSV lower and upper range to BGR for drawing rectangles
    lower_bgr = cv.cvtColor(np.uint8([[lower]]), cv.COLOR_HSV2BGR)[0][0]
    upper_bgr = cv.cvtColor(np.uint8([[upper]]), cv.COLOR_HSV2BGR)[0][0]

    # Train KNN with example pixels for each color
    training_data.append([(lower + upper) // 2])
    training_labels.append(color)

training_data = np.array(training_data)
training_labels = np.array(training_labels)

training_data = training_data.reshape((len(training_data), -1))

knn_classifier.fit(training_data, training_labels)

while True:
    _, imageFrame = webcam.read()
    hsvFrame = cv.cvtColor(imageFrame, cv.COLOR_BGR2HSV)
    hsv_data = hsvFrame.reshape((-1, 3))
    predicted_labels = knn_classifier.predict(hsv_data)

    predicted_labels = predicted_labels.reshape(imageFrame.shape[:2])

    for color, (lower, upper) in colors_to_detect.items():
        mask = cv.inRange(hsvFrame, np.array(lower), np.array(upper))
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv.contourArea(contour) > 100:
                x, y, w, h = cv.boundingRect(contour)
                cv.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 0, 0), 2)
                cv.putText(imageFrame, color, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    cv.imshow("Color Detection with KNN", imageFrame)

    if cv.waitKey(10) & 0xFF == ord('q'):
        webcam.release()
        cv.destroyAllWindows()
        break
