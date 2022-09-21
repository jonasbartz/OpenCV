import cv2

cap = cv2.VideoCapture('Patrick.mp4')

classificadorFace = cv2.CascadeClassifier('cascades\haarcascade_frontalface_default.xml')

amostra = 1
numeroAmostras = 1000

id = input('Digite seu identificador: ')
largura, altura = 220, 220

print("Captura as faces...")

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    facesDetectadas = classificadorFace.detectMultiScale(gray, scaleFactor = 1.5, minSize=(150,150))

    for (x, y, a, l) in facesDetectadas:
        cv2.rectangle(gray, (x, y), (x + l, y + a), (0, 0, 255), 2)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        imagemFace = cv2.resize(gray[y:y + a, x:x + l], (largura, altura))
        cv2.imwrite("fotos/pessoa." + str(id) + "." + str(amostra) + ".jpg", imagemFace)
        print("[foto " + str(amostra) + " capturada com sucesso]")
        amostra +=1

    cv2.imshow('Face',gray)
    cv2.waitKey(1)

    if (amostra >= numeroAmostras + 1):
        break

cap.release()
cv2.destroyAllWindows()