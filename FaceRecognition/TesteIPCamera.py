import cv2

while True:
    cap = cv2.VideoCapture('http://10.91.60.60/image/jpeg.cgi')
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()