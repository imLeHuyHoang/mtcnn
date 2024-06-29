import cv2 as cv
import numpy as np
import pickle
from keras_facenet import FaceNet
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

facenet = FaceNet()
face_embedding = np.load(
    "C:/Users/lehuy/OneDrive/Desktop/mtcnn/code/faces_embeddings_done.npz"
)
Y = face_embedding["arr_1"]

encoder = LabelEncoder()
encoder.fit(Y)
haarcascade = cv.CascadeClassifier(
    "C:/Users/lehuy/OneDrive/Desktop/mtcnn/code/haarcascade_frontalface_default.xml"
)
model = pickle.load(
    open("C:/Users/lehuy/OneDrive/Desktop/mtcnn/code/svm_model_160x160.pkl", "rb")
)
cap = cv.VideoCapture(0)

name_appear = []
frame_count = 0
detect_interval = 5

while cap.isOpened():
    _, frame = cap.read()

    frame_resized = cv.resize(frame, (640, 480))

    if frame_count % detect_interval == 0:
        gray_img = cv.cvtColor(frame_resized, cv.COLOR_BGR2GRAY)
        faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)

        for x, y, w, h in faces:
            img = frame_resized[y : y + h, x : x + w]
            img = cv.resize(img, (160, 160))
            img = np.expand_dims(img, axis=0)
            ypred = facenet.embeddings(img)
            face_proba = model.predict_proba(ypred)[0]
            max_proba_index = np.argmax(face_proba)
            max_proba = face_proba[max_proba_index]

            if max_proba > 0.4:
                face_name = model.classes_[max_proba_index]
                final_name = encoder.inverse_transform([face_name])
                if final_name not in name_appear:
                    name_appear.append(final_name)
                    with open(
                        "C:/Users/lehuy/OneDrive/Desktop/mtcnn/attendance.csv", "a"
                    ) as f:
                        f.write(str(final_name[0]) + "," + str(datetime.now()) + "\n")
            else:
                final_name = ["unknown"]

            cv.rectangle(frame_resized, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv.putText(
                frame_resized,
                str(final_name),
                (x, y),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

    cv.imshow("frame", frame_resized)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

    frame_count += 1

cap.release()
cv.destroyAllWindows()
