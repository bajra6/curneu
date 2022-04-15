import cv2
import numpy as np

cap = cv2.VideoCapture("DS-IQ-002-ObjectDetect-Video.mp4")

prev = np.empty([0, 0, 0])
while True:
    ret, origFrame = cap.read()
    frame = cv2.cvtColor(origFrame, cv2.COLOR_BGR2GRAY)
    roi = frame[200: 620, 500: 1000]
    if prev.shape != (0, 0, 0):
        diff = cv2.absdiff(prev, roi)
        ret, thresh = cv2.threshold(diff, 75, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        blurred = cv2.blur(dilated, (5, 5))
        contours, hierarchy = cv2.findContours(
            blurred.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours = [
            contour for contour in contours if cv2.contourArea(contour) > 100]
        contoursLen = str(len(contours))

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(origFrame, (x+500, y+200),
                          (x+w+500, y+h+200), (0, 255, 0), 2)
            cv2.putText(origFrame, contoursLen, (1200, 700),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Frame", origFrame)
    prev = frame[200: 620, 500: 1000]

    key = cv2.waitKey(30)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
