from turtle import width

from venv import create

import numpy as np

import cv2

import math

import time

total_frames = []

array1 = []

array2 = []

frames = 0

f = 1

fpr = 0

revolutions = 0

rounded_rpm = 0

timer = 0

counter = 0  # Counter for revolutions

video_file = r"1.MOV"


# Reduces the size of the video displayed

def rescaleFrame(frame, scale=0.5):
    width = int(frame.shape[1] * scale)

    height = int(frame.shape[0] * scale)

    dimesions = (width, height)

    return cv2.resize(frame, dimesions, interpolation=cv2.INTER_AREA)


# Tracks the moving blade

tracker = cv2.TrackerKCF_create()

cap = cv2.VideoCapture(video_file)

# Gets the fps of the video

fps = int(cap.get(cv2.CAP_PROP_FPS))

# Asks the question
td = int(input("Duration of the video (in seconds) : "))
ans = input("Is the blade in the video (1) symmetrical or (2) asymmetrical? ")

# If statement on symmetrical and asymmetrical lines

success, img = cap.read()

frame_resized = rescaleFrame(img)

roi = cv2.selectROI(frame_resized, False)

success = tracker.init(frame_resized, roi)

object_detector = cv2.createBackgroundSubtractorMOG2()

# Define text parameters

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
color = (256, 256, 256)  # BGR color (white in this case)
thickness = 1

frame_limit = (((round(fps, -1)) - 1) * td) - 5


def calc_rpm(ans, fpr, fps):
    if ans == "1":
        # Calculates frames per rotation for symmetric

        for i in range(len(array2) - 2):  # Iterate until the second last element
            fpr += array2[i + 2] - array2[i]

        fpr /= (len(array2) - 2)  # Divide the sum by the number of differences

        # fpr = ((array2[2]-array2[0])+(array2[3]-array2[1])+(array2[4]-array2[2])+(array2[5]-array2[3]))/4

    elif ans == "2":

        for i in range(len(array2) - 1):  # Iterate until the last element
            fpr += array2[i + 1] - array2[i]

        fpr /= (len(array2) - 1)  # Divide the sum by the number of differences

        # fpr = ((array2[1]-array2[0])+(array2[2]-array2[1])+(array2[3]-array2[2])+(array2[4]-array2[3]))/4

    rounded_fpr = "{:.1f}".format(fpr)

    # Finds revolutions per minute

    rpm = float((fps / fpr) * 60)

    rounded_rpm = "{:.2f}".format(rpm)

    return fps, rounded_fpr, rounded_rpm


while True:
    text = f"Revolutions: {revolutions} | RPM: {rounded_rpm} | Elapsed Time : {timer}"

    # Determine the text size
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

    # Position the text in the top left corner
    text_x = 10
    text_y = text_size[1] + 10

    frames += 1

    success, img = cap.read()

    frame_resized = rescaleFrame(img)

    success, roi = tracker.update(frame_resized)

    if (frames >= frame_limit and round(fps, -1) == 60) or (frames >= frame_limit and round(fps, -1) == 30):
        break

    # prints out every minute

    if (frames % 60 == 0 and round(fps, -1) == 60) or (frames % 30 == 0 and round(fps, -1) == 30):
        timer += 1
        _, _, rounded_rpm = calc_rpm(ans, fpr, fps)

    if (frames % 3600 == 0 and round(fps, -1) == 60):
        fps, rounded_fpr, rounded_rpm = calc_rpm(ans, fpr, fps)
        print(
            f"In this {fps} fps video, the blade completes a full rotation every {rounded_fpr} frames. The blade has {rounded_rpm} revolutions per minute. Total revolution is {revolutions}"
        )

    if (frames % 1800 == 0 and round(fps, -1) == 30):
        print(array2)
        calc_rpm(ans, fpr, fps)

    if success:

        (x, y, w, h) = [int(v) for v in roi]

        cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2, 1)

        height_a = y - 25

        width_a = y - 20

        new_height = x + 25

        new_width = x + 20

        area1 = [(width_a, height_a), (new_width, height_a), (new_width, new_height), (width_a, new_height)]

        # Make a copy of the frame before drawing the unselected box
        frame_with_box = frame_resized.copy()

        # Draw the unselected box on the copy frame
        cv2.polylines(frame_with_box, [np.array(area1, np.int32)], True, (0, 220, 0), 2)

        check_in_roi = cv2.pointPolygonTest(np.array(area1, np.int32), (x, y), False)

        if check_in_roi == 1:
            total_frames.append(frames)

        for i in total_frames:

            if i == f:

                array1.append(i)

                f += 1

            elif i > f:

                array2.append(i)

                f = i + 1

                counter += 1

                if counter == 2:
                    revolutions += 1

                    counter = 0

    # cv2.imshow("Frame", frame_resized)

    # cv2.imshow("ROI", roi)

    # Draw the text on the image
    cv2.putText(frame_resized, text, (text_x, text_y), font, font_scale, color, thickness)

    # Display the image
    cv2.imshow("Text Image", frame_resized)

    if round(fps, -1) == 30:
        key = cv2.waitKey(60)
    elif round(fps, -1) == 60:
        key = cv2.waitKey(30)

    if key == 27:
        break

fps, rounded_fpr, rounded_rpm = calc_rpm(ans, fpr, fps)
print(f"In this {fps} fps video, the blade completes a full rotation every {rounded_fpr} frames. The blade has {rounded_rpm} revolutions per minute. Total revolution is {revolutions}")

cap.release()

cv2.destroyAllWindows()
