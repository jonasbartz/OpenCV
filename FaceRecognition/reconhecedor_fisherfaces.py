import cv2

classificadorFace = cv2.CascadeClassifier('cascades\haarcascade_frontalface_default.xml')
reconhecedor = cv2.face.FisherFaceRecognizer_create()
reconhecedor.read("classificadorfisherface.yml")

largura, altura = 220, 220
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

cap = cv2.VideoCapture('ParteEquipe.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    #print(frame)

    #gray = frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    facesDetectadas = classificadorFace.detectMultiScale(gray, scaleFactor=2, minSize=(20,20), minNeighbors=5)

    for (x, y, a, l) in facesDetectadas:
        imagemFace = cv2.resize(gray[y:y + a, x:x + l], (largura, altura))
        cv2.rectangle(gray, (x, y), (x + l, y + a), (255, 0, 255), 2)
        id, confianca = reconhecedor.predict(imagemFace)
        nome = ""
        if id == 1:
            nome = 'Jonas'
        elif id == 2:
            nome = 'Bruno'
        elif id == 3:
            nome = 'Patrick'

        #cv2.putText(gray, str(id), (x,y + (a + 30)), font, 2, (0,0,255))
        cv2.putText(gray, nome, (x, y + (a + 30)), font, 2, (255, 0, 255))
        cv2.putText(gray, str(confianca), (x, y + (a + 50)), font, 1, (255, 0, 255))

    cv2.imshow('frame',gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()