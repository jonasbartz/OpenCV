import cv2

classificadorFace = cv2.CascadeClassifier('cascades\haarcascade_frontalface_default.xml')
reconhecedor = cv2.face.EigenFaceRecognizer_create()
reconhecedor.read("classificadorEigen.yml")

largura, altura = 220, 220
#font = cv2.FONT_HERSHEY_TRIPLEX
font = cv2.FONT_HERSHEY_PLAIN

amostra = 1

while True:
    cap = cv2.VideoCapture('http://10.91.60.60/image/jpeg.cgi')

    ret, frame = cap.read()

    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    facesDetectadas = classificadorFace.detectMultiScale(gray, scaleFactor=1.3, minSize=(30, 30), minNeighbors=5)

#    for (x, y, a, l) in facesDetectadas:
#        cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 255), 2)

    for (x, y, a, l) in facesDetectadas:
        imagemFace = cv2.resize(gray[y:y + a, x:x + l], (largura, altura))

        cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 255), 2)
        id, confianca = reconhecedor.predict(imagemFace)

        nome = ""
        if id == 1:
            nome = 'Jonas'

        cv2.putText(frame, nome, (x, y + (a + 30)), font, 2, (0, 0, 255))
        cv2.putText(frame, str(confianca), (x, y + (a + 50)), font, 1, (0, 0, 255))

    cv2.imshow('Face recognition',frame)

    #Encerra reconhecimento
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()