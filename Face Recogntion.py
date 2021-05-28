import cv2
import numpy as np
import face_recognition
import os

path = 'Images'
images = []
classname = []
mylist = os.listdir(path)
print(mylist)
for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classname.append(os.path.splitext(cl)[0])
print(classname)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img =cap.read()
    imgS = cv2.resize(img,(0,0),None,1,1)
    img = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facecurframe = face_recognition.face_locations(imgS)
    encodecurframe = face_recognition.face_encodings(imgS,facecurframe)

    for encodeface,faceloc in zip(encodecurframe,facecurframe):
        matches = face_recognition.compare_faces(encodeListKnown, encodeface)
        facedis = face_recognition.face_distance(encodeListKnown, encodeface)
        #print(facedis)
        matchIndex = np.argmin(facedis)

        if matches[matchIndex]:
            name = classname[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceloc
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)


    cv2.imshow('Face Recognition',img)
    cv2.waitKey(1)
