import cv2

cap = cv2.VideoCapture('ParteEquipe.mp4')

classificadorFace = cv2.CascadeClassifier('cascades\haarcascade_frontalface_default.xml')

while(cap.isOpened()):
    ret, frame = cap.read()
    #print(frame)

    #gray = frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    facesDetectadas = classificadorFace.detectMultiScale(gray, scaleFactor=2, minSize=(30,30), minNeighbors=5)

    for (x, y, a, l) in facesDetectadas:
        cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 255), 2)

    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()