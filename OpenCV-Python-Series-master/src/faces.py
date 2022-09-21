import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
#eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
#smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('recognizers/face-trainner.yml')

#recognizer = cv2.face.FisherFaceRecognizer_create()
#recognizer.read('recognizers/classificadorfisherface.yml')

#recognizer = cv2.face.EigenFaceRecognizer_create(num_components=50)
#recognizer.read('recognizers/classificadorEigen.yml')

width_d, height_d = 280, 280

labels = {"person_name": 1}
with open("pickles/face-labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

while(True):
    cap = cv2.VideoCapture('http://10.91.60.60/image/jpeg.cgi')

    ret, frame = cap.read()

    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        #print(x,y,w,h)
        #roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
        #roi_color = frame[y:y+h, x:x+w]

        roi_gray = cv2.resize(gray[y:y + h, x:x + w], (width_d, height_d))
        roi_color = cv2.resize(frame[y:y + h, x:x + w], (width_d, height_d))

        # recognize? deep learned model predict keras tensorflow pytorch scikit learn
        id_, conf = recognizer.predict(roi_gray)

        #if conf>=6000 and conf <= 8000:
        #if conf >= 2000 and conf <= 4000:
        if conf >= 45 and conf <= 85:
            #print(5: #id_)
            #print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2

            name = name +' - '+ str(conf)

            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
            #cv2.putText(frame, 'Teste', (x, y), font, 1, color, stroke, cv2.LINE_AA)
        #EndIF

        img_item = "7.png"
        cv2.imwrite(img_item, roi_color)

        color = (255, 0, 0) #BGR 0-255
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
        #subitems = smile_cascade.detectMultiScale(roi_gray)
        #for (ex,ey,ew,eh) in subitems:
        #	cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
