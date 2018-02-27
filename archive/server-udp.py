import socket
import numpy
import time
import cv2

capture = cv2.VideoCapture("udp://108.179.161.77:9999/?overrun_nonfatal=True&fifo_size=1000000")

while True:
    success, frame = capture.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()