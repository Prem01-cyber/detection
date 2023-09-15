import cv2 as cv
import numpy as np
from face_recognition import load_image_file, face_locations, face_encodings, compare_faces, face_distance

# Load the image and convert it from BGR to RGB
image = load_image_file("./assets/3.jpg")
rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
image2 = load_image_file("./assets/4.jpg")
rgb2 = cv.cvtColor(image2, cv.COLOR_BGR2RGB)

# Detect the coordinates of the faces
boxes = face_locations(rgb, model="hog")
boxes2 = face_locations(rgb2, model="hog")

# Compute the facial embeddings
encodings = face_encodings(rgb, boxes)
encodings2 = face_encodings(rgb2, boxes2)


# Loop over the encodings
for encoding in encodings:
    matches = compare_faces(encodings2, encoding)
    distance = face_distance(encodings2, encoding)
    print(distance, matches)
    # Check if at least one match was found
    if True in matches:
        print("Match found")
    else:
        print("Match not found")

# Loop over the encodings
for encoding in encodings2:
    matches = compare_faces(encodings, encoding)
    distance = face_distance(encodings, encoding)
    print(distance, matches)
    # Check if at least one match was found
    if True in matches:
        print(f"{matches} {round(distance[0],2)}")
    else:
        print("Match not found")

# Loop over the bounding boxes
for ((top, right, bottom, left), encoding) in zip(boxes, encodings):
    # Draw the predicted face name on the image
    cv.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    y = top - 15 if top - 15 > 15 else top + 15
    cv.putText(image, "Face", (left, y), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

# Loop over the bounding boxes
for ((top, right, bottom, left), encoding) in zip(boxes2, encodings2):
    # Draw the predicted face name on the image
    cv.rectangle(image2, (left, top), (right, bottom), (0, 255, 0), 2)
    y = top - 15 if top - 15 > 15 else top + 15
    cv.putText(image2, "Face", (left, y), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

# Show the image
cv.imshow("Image", image)
cv.imshow("Image2", image2)

cv.waitKey(0)
cv.destroyAllWindows()