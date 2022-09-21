import cv2

classificadorFace = cv2.CascadeClassifier('cascades\haarcascade_frontalface_default.xml')

largura, altura = 220, 220
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
amostra = 1

Nome = input("Nome da foto:")

cap = cv2.VideoCapture('http://10.91.60.60/image/jpeg.cgi')
ret, frame = cap.read()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

facesDetectadas = classificadorFace.detectMultiScale(gray, scaleFactor=1.2, minSize=(30, 30), minNeighbors=5)

for (x, y, a, l) in facesDetectadas:
    cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 255), 2)
    crop = frame[y:y+a, x:x+l]
    cv2.imwrite("fotos/" + Nome + ".jpg", crop)

face = cv2.imread("fotos/" + Nome + ".jpg",0)

cv2.imshow('Face recortada', face)


cap.release()
cv2.destroyAllWindows()