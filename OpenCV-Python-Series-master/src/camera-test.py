import numpy as np
import cv2

while(True):
    cap = cv2.VideoCapture('http://10.91.60.60/image/jpeg.cgi')

    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        continue

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    cv2.imshow('gray',gray)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()