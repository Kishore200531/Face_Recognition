import cv2, numpy, os
haar_file = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_file)
datasets = "Dataset"
print("Training.....")
(images,labels,names,id) = ([],[],{},0)

for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + "/" + filename
            label = id
            images.append(cv2.imread(path,0))
            labels.append(int(label))
        id +=1

(images,labels) = [numpy.array(lis) for lis in [images,labels]]
(width,height) = (400,400)
model = cv2.face.LBPHFaceRecognizer_create() #model = cv2.face.FisherFaceRecognizer_create()
model.train(images,labels)

webcam = cv2.VideoCapture(0)
cnt = 0

while True:
    _,img = webcam.read()
    grayimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayimg,1.2,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        face = grayimg[y:y+h,x:x+w]
        face_resize = cv2.resize(face,(width,height))

        prediction = model.predict(face_resize)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        if prediction[1] < 900:
            cv2.putText(img,"%s - %.0f" % (names[prediction[0]],prediction[1]),(x-20,y-20),cv2.FONT_HERSHEY_PLAIN,2,(0,0,0))
            print(names[prediction[0]])
            cnt=0
        else:
            cnt+=1
            cv2.putText(img,"Unknown",(x-20,y-20),cv2.FONT_HERSHEY_PLAIN,2,(0,0,0))
            if(cnt>100):
                print("Unknown Person")
                cv2.imwrite("Unknown.jpg",img)
                cnt=0

    cv2.imshow("FaceRecognition",img)
    key = cv2.waitKey(10)
    if key == ord("j"):
        break

webcam.release()
cv2.destroyAllWindows()
