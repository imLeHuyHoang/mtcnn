import os
import cv2 as cv
import numpy as np
import tensorflow as tf
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import threading
import tkinter as tk
from tkinter import ttk

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class FACELOADING:
    def __init__(self, directory):
        self.directory = directory
        self.target_size = (160, 160)
        self.X, self.Y = [], []
        self.detector = MTCNN()

    def extract_face(self, filename):
        img = cv.cvtColor(cv.imread(filename), cv.COLOR_BGR2RGB)
        x, y, w, h = self.detector.detect_faces(img)[0]["box"]
        return cv.resize(img[abs(y) : y + h, abs(x) : x + w], self.target_size)

    def load_faces(self, dir):
        return [
            self.extract_face(dir + im_name)
            for im_name in os.listdir(dir)
            if self.extract_face(dir + im_name) is not None
        ]

    def load_classes(self):
        for sub_dir in os.listdir(self.directory):
            path = self.directory + "/" + sub_dir + "/"
            FACES = self.load_faces(path)
            self.X.extend(FACES)
            self.Y.extend([sub_dir] * len(FACES))
        return np.asarray(self.X), np.asarray(self.Y)


def start_progress(progress, root):
    faceloading = FACELOADING("C:/Users/lehuy/OneDrive/Desktop/mtcnn/dataset")
    X, Y = faceloading.load_classes()
    progress["value"] = 50
    root.update_idletasks()

    embedder = FaceNet()
    EMBEDDED_X = np.asarray(
        [
            embedder.embeddings(np.expand_dims(img.astype("float32"), axis=0))[0]
            for img in X
        ]
    )
    progress["value"] = 100
    root.update_idletasks()

    np.savez_compressed("C:/Users/lehuy/OneDrive/Desktop/mtcnn/code/faces_embeddings_done.npz", EMBEDDED_X, Y)

    encoder = LabelEncoder()
    Y = encoder.fit_transform(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(
        EMBEDDED_X, Y, shuffle=True, test_size=0.2, random_state=42
    )

    model = SVC(kernel="linear", probability=True).fit(X_train, Y_train)

    print("accuracy: ", accuracy_score(Y_test, model.predict(X_test)))
    print("confusion matrix: ", confusion_matrix(Y_test, model.predict(X_test)))

    with open("C:/Users/lehuy/OneDrive/Desktop/mtcnn/code/svm_model_160x160.pkl", "wb") as f:
        pickle.dump(model, f)


root = tk.Tk()
root.title("Progress Bar")

progress = ttk.Progressbar(root, length=100, mode="determinate")
progress.pack(pady=10)

start_button = ttk.Button(
    root,
    text="Start",
    command=lambda: threading.Thread(
        target=start_progress, args=(progress, root)
    ).start(),
)
start_button.pack(pady=10)

back_button = ttk.Button(root, text="Back", command=root.quit)
back_button.pack(pady=10)


root.mainloop()
