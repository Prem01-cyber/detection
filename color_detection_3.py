import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans
from webcolors import rgb_to_name
import time

# Initialize the webcam
webcam = cv.VideoCapture(0)

# Number of clusters (colors) to detect
num_clusters = 5

# Define the size of the sliding window (kernel)
kernel_size = 30

# Define the minimum confidence level for color detection
min_confidence = 1000  # Adjust this value as needed

# Define the duration for color detection in seconds
detection_duration = 10  # Adjust the duration as needed

# Define the interval for displaying colors in milliseconds
display_interval = 1000  # Display every 1 second

start_time = time.time()
detected_colors = []

def rgb_to_color_name(rgb_color):
    try:
        color_name = rgb_to_name(rgb_color, spec='css3')
        return color_name
    except ValueError:
        return 'Unknown'

display_start_time = time.time()

while True:
    _, imageFrame = webcam.read()

    # Define the step size for sliding the window
    step = 10

    for x in range(0, imageFrame.shape[1] - kernel_size, step):
        for y in range(0, imageFrame.shape[0] - kernel_size, step):
            # Extract the kernel window
            roi = imageFrame[y:y+kernel_size, x:x+kernel_size]

            # Calculate the mean color within the kernel window
            mean_color = np.mean(roi, axis=(0, 1)).astype(int)

            # Convert RGB color to color name using webcolors with fallback
            color_name = rgb_to_color_name(tuple(mean_color))

            # Calculate color confidence (based on pixel intensity)
            confidence = np.sum(roi) / (3 * kernel_size * kernel_size)

            # Store detected color and position if confidence is above the threshold
            if confidence >= min_confidence and color_name != 'Unknown':
                detected_colors.append((x, y, kernel_size, kernel_size, color_name))

    # Check if the detection duration has elapsed
    if time.time() - start_time >= detection_duration:
        # Reset the start time for color display
        start_time = time.time()

        # Perform K-Means clustering on the detected colors if there are enough samples
        if len(detected_colors) >= num_clusters:
            detected_colors = np.array([c[:3] for c in detected_colors])
            kmeans = KMeans(n_clusters=num_clusters)
            kmeans.fit(detected_colors)
            clustered_colors = kmeans.cluster_centers_.astype(int)

            # Create an image to display the clustered colors
            color_display = np.zeros((100, 100, 3), dtype=np.uint8)
            for i, color in enumerate(clustered_colors):
                color = tuple(color)
                cv.rectangle(color_display, (i * 20, 0), ((i + 1) * 20, 100), color, -1)

            cv.imshow('Detected Colors', color_display)

    # Check if it's time to update the display
    if time.time() - display_start_time >= display_interval:
        display_start_time = time.time()
        cv.imshow('Video Feed', imageFrame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        webcam.release()
        cv.destroyAllWindows()
        break
