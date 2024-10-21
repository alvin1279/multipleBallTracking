import cv2
import numpy as np
import imutils
import math
from collections import defaultdict
import hsvMaskUtility as hlpr
from scipy.spatial import distance as dist
from centroidClass import CentroidTracker

# Initialize tracking-related variables
speeds = {}
last_speeds = defaultdict(lambda: [])

# Speed smoothing window size
SPEED_WINDOW_SIZE = 5

# HSV range for mask
lower = (26, 42, 167)
upper = (179, 255, 255)

# Capture video
vs = cv2.VideoCapture('Samples/vid1.mp4')
if not vs.isOpened():
    raise IOError("Cannot open video file")

fps = vs.get(cv2.CAP_PROP_FPS)

# Initialize centroid tracker
ct = CentroidTracker()

# Function to calculate Euclidean distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

# Function to draw tracking information on the frame
def draw_tracking_info(frame, blank_image, cnts, centroid, objectID, angle_degrees, speeds):
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if (int(x + w / 2), int(y + h / 2)) == centroid:
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Extend arrow endpoint based on angle
            xext = int(centroid[0] + 100 * math.cos(math.radians(angle_degrees)))
            yext = int(centroid[1] + 100 * math.sin(math.radians(angle_degrees)))

            # Draw arrow for movement direction
            cv2.arrowedLine(frame, centroid, (xext, yext), (0, 255, 0), 2, tipLength=0.2)

            # Display object ID and speed
            cv2.putText(frame, f"ID {objectID} Speed: {speeds[objectID]:.2f} px/sec", 
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Draw all this in blank image
            cv2.rectangle(blank_image, (x, y), (x + w, y + h), (255, 255, 255), -1)
            cv2.arrowedLine(blank_image, centroid, (xext, yext), (0, 255, 0), 2, tipLength=0.2)
            cv2.putText(blank_image, f"ID {objectID} Speed: {speeds[objectID]:.2f} px/sec", 
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Main loop for video processing
while True:
    ret, frame = vs.read()
    if not ret:
        break

    # Preprocess the frame
    frame = imutils.resize(frame, width=900)
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create blank image same as frame
    blank_image = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)

    # Get mask for object detection
    mask = hlpr.GetMask(hsvImage, lower, upper, 3)
    cv2.imshow("mask", mask)

    # Find contours
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(contours)

    # Extract centroids from contours
    inputCentroids = []
    for c in cnts:
        if cv2.contourArea(c) < 100:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        centroid = (int(x + w / 2), int(y + h / 2))
        inputCentroids.append(centroid)

    # Update the tracker with new centroids
    objects = ct.update(inputCentroids)

    # Process each tracked object
    for objectID, obj in ct.objects.items():
        centroid = obj.centroid
        if len(obj.centroid_history) > 1:
            # Calculate displacement and speed
            previous_centroids = obj.centroid_history[:-1]
            # previous_centroid = obj.centroid_history[-2]
            previous_centroid = tuple(map(lambda x: int(sum(x) / len(x)), zip(*previous_centroids)))

            displacement = calculate_distance(previous_centroid, centroid)
            time_interval = 1 / fps
            current_speed = displacement / time_interval

            # Add the current speed to the smoothing window
            last_speeds[objectID].append(current_speed)
            if len(last_speeds[objectID]) > SPEED_WINDOW_SIZE:
                last_speeds[objectID].pop(0)

            # Smoothed speed (moving average)
            smoothed_speed = sum(last_speeds[objectID]) / len(last_speeds[objectID])
            if smoothed_speed < 30:  # Speed threshold to filter noise
                smoothed_speed = 0
            speeds[objectID] = smoothed_speed

            # Calculate movement direction (angle)
            delta_x = centroid[0] - previous_centroid[0]
            delta_y = centroid[1] - previous_centroid[1]
            if delta_x != 0 or delta_y != 0:
                angle = math.atan2(delta_y, delta_x)
                angle_degrees = math.degrees(angle)
            else:
                angle_degrees = 0
        else:
            speeds[objectID] = 0
            angle_degrees = 0

        # Normalize the angle to [0, 360)
        if angle_degrees < 0:
            angle_degrees += 360

        # Draw tracking information
        draw_tracking_info(frame, blank_image, cnts, centroid, objectID, angle_degrees, speeds)

    # Show frame
    cv2.imshow("Frame", frame)
    cv2.imshow("blank_image", blank_image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Cleanup
vs.release()
cv2.destroyAllWindows()
