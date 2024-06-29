import tkinter as tk
import cv2
import threading
import tkinter as tk
from PIL import Image, ImageTk
import cv2 as cv
import numpy as np
import pickle
from keras_facenet import FaceNet
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import subprocess
import os
from tkinter import messagebox
import sqlite3
from tqdm import tqdm
import time
from tkinter import ttk


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Phần mềm điểm danh sinh viên")
        self.root.geometry("800x500")
        self.root.resizable(False, False)

        self.window1 = None
        self.window2 = None
        self.window3 = None
        self.window4 = None
        self.window5 = None
        self.window6 = None

        self.label = tk.Label(
            self.root, text="Phần mềm điểm danh sinh viên", font=("Arial", 20)
        )
        self.label.grid(row=0, column=0, columnspan=3, pady=20)

        self.button1 = tk.Button(
            self.root, text="Điểm danh", font=("Arial", 15), command=self.open_window1
        )
        self.button1.grid(row=1, column=0, padx=20, pady=20)

        self.button2 = tk.Button(
            self.root,
            text="Thêm sinh viên",
            font=("Arial", 15),
            command=self.open_window2,
        )
        self.button2.grid(row=1, column=1, padx=20, pady=20)

        self.button3 = tk.Button(
            self.root,
            text="Xem danh sách sinh viên",
            font=("Arial", 15),
            command=self.open_window3,
        )
        self.button3.grid(row=2, column=1, padx=20, pady=20)

        self.button4 = tk.Button(
            self.root,
            text="Xóa sinh viên",
            font=("Arial", 15),
            command=self.open_window4,
        )
        self.button4.grid(row=2, column=0, padx=20, pady=20)

        self.button5 = tk.Button(
            self.root, text="Train model", font=("Arial", 15), command=self.open_window5
        )
        self.button5.grid(row=3, column=0, padx=20, pady=20)

        self.button6 = tk.Button(
            self.root,
            text="Xuất file điểm danh",
            font=("Arial", 15),
            command=self.open_window6,
        )
        self.button6.grid(row=3, column=1, padx=20, pady=20)

    def open_window1(self):
        self.root.withdraw()
        subprocess.call(["python", "C:/Users/lehuy/OneDrive/Desktop/mtcnn/Nhandien.py"])
        messagebox.showinfo("Thông báo", "đã thực hiện lệnh được chọn")
        self.root.deiconify()

    def capture_images(self, ID):
        dataset_path = "C:\\Users\\lehuy\\OneDrive\\Desktop\\mtcnn\\dataset"
        id_folder = os.path.join(dataset_path, str(ID))
        if not os.path.exists(id_folder):
            os.makedirs(id_folder)

        images = os.listdir(id_folder)
        num_images = len(images)

        cap = cv2.VideoCapture(0)
        for i in range(num_images, num_images + 10):
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(os.path.join(id_folder, f"{i}.jpg"), frame)
        cap.release()

    def open_window2(self):
        if self.window2 is None:
            self.window2 = tk.Toplevel(self.root)
            self.window2.title("Thêm sinh viên")
            self.window2.geometry("800x500")
            self.window2.resizable(False, False)
            self.root.withdraw()

            name_label = tk.Label(self.window2, text="Tên sinh viên:")
            name_label.grid(row=0, column=0)
            name_entry = tk.Entry(self.window2)
            name_entry.grid(row=0, column=1)

            id_label = tk.Label(self.window2, text="Mã sinh viên:")
            id_label.grid(row=1, column=0)
            id_entry = tk.Entry(self.window2)
            id_entry.grid(row=1, column=1)

            class_name_label = tk.Label(self.window2, text="Lớp:")
            class_name_label.grid(row=2, column=0)
            class_name_entry = tk.Entry(self.window2)
            class_name_entry.grid(row=2, column=1)

            contact_label = tk.Label(self.window2, text="Số điện thoại:")
            contact_label.grid(row=3, column=0)
            contact_entry = tk.Entry(self.window2)
            contact_entry.grid(row=3, column=1)

            andress_label = tk.Label(self.window2, text="Địa chỉ:")
            andress_label.grid(row=4, column=0)
            andress_entry = tk.Entry(self.window2)
            andress_entry.grid(row=4, column=1)

            def add_student():
                dataset_path = "C:\\Users\\lehuy\\OneDrive\\Desktop\\mtcnn\\dataset"
                name = name_entry.get()
                ID = id_entry.get()

                id_folder = os.path.join(dataset_path, str(ID))
                if os.path.exists(id_folder):
                    messagebox.showinfo("Thông báo", "ID đã tồn tại.")
                    self.capture_images(ID)
                else:
                    messagebox.showinfo("Thông báo", "Tạo ID mới...")
                    os.makedirs(id_folder)
                    self.capture_images(ID)

                conn = sqlite3.connect("students.db")
                c = conn.cursor()

                c.execute(
                    "INSERT INTO students (id, name, class, contact, address) VALUES (?, ?, ?, ?, ?)",
                    (
                        ID,
                        name,
                        class_name_entry.get(),
                        contact_entry.get(),
                        andress_entry.get(),
                    ),
                )

                conn.commit()
                conn.close()

                messagebox.showinfo("Thông báo", "Chụp ảnh thành công.")
                self.window2.destroy()
                self.window2 = None
                self.root.deiconify()

            add_button = tk.Button(
                self.window2, text="Thêm sinh viên", command=add_student
            )
            add_button.grid(row=5, column=0, columnspan=2)
            button_back = tk.Button(
                self.window2, text="Quay lại", command=self.back_to_main1
            )
            button_back.grid(row=6, column=0, columnspan=2)

    def back_to_main1(self):
        self.window2.destroy()
        self.window2 = None
        self.root.deiconify()

    def open_window3(self):
        if self.window3 is None:
            self.window3 = tk.Toplevel(self.root)
            self.window3.title("Xem danh sách sinh viên")
            self.window3.geometry("800x500")
            self.window3.resizable(False, False)
            self.root.withdraw()

            conn = sqlite3.connect("students.db")
            c = conn.cursor()

            c.execute("SELECT * FROM students")
            students = c.fetchall()

            for i, student in enumerate(students):
                id, name, class_name, contact, address = student
                tk.Label(
                    self.window3,
                    text=f"{i + 1}. {name} - {id} - {class_name} - {contact} - {address}",
                ).pack()

            conn.close()

            back_button = tk.Button(
                self.window3, text="Quay lại", command=self.back_to_main2
            )
            back_button.pack()

    def back_to_main2(self):
        self.window3.destroy()
        self.window3 = None
        self.root.deiconify()

    def open_window4(self):
        if self.window4 is None:
            self.window4 = tk.Toplevel(self.root)
            self.window4.title("Xóa sinh viên")
            self.window4.geometry("800x500")
            self.window4.resizable(False, False)
            self.root.withdraw()

            id_label = tk.Label(self.window4, text="Mã sinh viên:")
            id_label.grid(row=0, column=0)
            id_entry = tk.Entry(self.window4)
            id_entry.grid(row=0, column=1)

            def delete_student():
                conn = sqlite3.connect("students.db")
                c = conn.cursor()

                c.execute("DELETE FROM students WHERE id = ?", (id_entry.get(),))

                conn.commit()
                conn.close()

                dataset_path = "C:\\Users\\lehuy\\OneDrive\\Desktop\\mtcnn\\dataset"
                id_folder = os.path.join(dataset_path, id_entry.get())
                if os.path.exists(id_folder):
                    for file in os.listdir(id_folder):
                        os.remove(os.path.join(id_folder, file))
                    os.rmdir(id_folder)

                messagebox.showinfo("Thông báo", "Xóa sinh viên thành công.")
                self.window4.destroy()
                self.window4 = None
                self.root.deiconify()

            delete_button = tk.Button(
                self.window4, text="Xóa sinh viên", command=delete_student
            )
            delete_button.grid(row=1, column=0, columnspan=2)

            back_button = tk.Button(
                self.window4, text="Quay lại", command=self.back_to_main3
            )
            back_button.grid(row=2, column=0, columnspan=2)

    def back_to_main3(self):
        self.window4.destroy()
        self.window4 = None
        self.root.deiconify()

    def open_window5(self):
        self.root.withdraw()
        subprocess.call(
            ["python", "C:/Users/lehuy/OneDrive/Desktop/mtcnn/code/Trainmodel.py"]
        )

        messagebox.showinfo("Thông báo", "đã thực hiện lệnh được chọn")
        self.root.deiconify()

    def back_to_main4(self):
        if self.window5 is not None:
            self.window5.destroy()
            self.window5 = None
        self.root.deiconify()

    def back_to_main5(self):
        if self.window5 is not None:
            self.window5.destroy()
            self.window5 = None
        if self.window6 is not None:
            self.window6.destroy()
            self.window6 = None
        self.root.deiconify()

    def open_window6(self):
        if self.window6 is None:
            self.window6 = tk.Toplevel(self.root)
            self.window6.title("Xuất file điểm danh")
            self.window6.geometry("800x500")
            self.window6.resizable(False, False)
            self.root.withdraw()

            export_today_button = tk.Button(
                self.window6,
                text="Xuất file điểm danh trong 1 ngày",
                command=self.export_today,
            )
            export_today_button.pack()

            export_all_button = tk.Button(
                self.window6,
                text="Xuất file điểm danh toàn bộ",
                command=self.export_all,
            )
            export_all_button.pack()

            analysis_button = tk.Button(
                self.window6,
                text="Đếm số buổi đi học của sinh viên",
                command=self.analysis,
            )
            analysis_button.pack()

            back_button = tk.Button(
                self.window6, text="Quay lại", command=self.back_to_main5
            )
            back_button.pack()

    def export_today(self):

        if self.window5 is None:
            self.window5 = tk.Toplevel(self.window6)
            self.window5.title("Xuất file điểm danh trong 1 ngày")
            self.window5.geometry("800x500")
            self.window5.resizable(False, False)
            self.window6.withdraw()

            date_label = tk.Label(self.window5, text="Ngày:")
            date_label.grid(row=0, column=0)
            date_entry = tk.Entry(self.window5)
            date_entry.grid(row=0, column=1)

            def export_today():
                with open("attendance.csv", "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        id, date_time = line.strip().split(",")
                        date = date_time.split(" ")[0]
                        if date == date_entry.get():
                            print(f"{id} - {date}")

                    with open(f"attendance{date_entry.get()}.csv", "w") as f:
                        for line in lines:
                            if (
                                line.strip().split(",")[1].split(" ")[0]
                                == date_entry.get()
                            ):
                                f.write(line)

            export_button = tk.Button(
                self.window5, text="Xuất file điểm danh", command=export_today
            )
            export_button.grid(row=1, column=0, columnspan=2)

            back_button = tk.Button(
                self.window5, text="Quay lại", command=self.back_to_main5
            )
            back_button.grid(row=2, column=0, columnspan=2)

    def export_all(self):
        with open("attendance.csv", "r") as f:
            lines = f.readlines()
            for line in lines:
                id, date = line.strip().split(",")
                tk.Label(self.window6, text=f"{id} - {date}").pack()

        with open("attendance_all.csv", "w") as f:
            for line in lines:
                f.write(line)

        messagebox.showinfo("Thông báo", "Xuất file điểm danh toàn bộ thành công.")

    def analysis(self):
        if self.window5 is None:
            self.window5 = tk.Toplevel(self.window6)
            self.window5.title("Đếm số buổi đi học của toàn bộ sinh viên")
            self.window5.geometry("800x500")
            self.window5.resizable(False, False)
            self.window6.withdraw()

            conn = sqlite3.connect("students.db")
            c = conn.cursor()

            c.execute("SELECT * FROM students")
            students = c.fetchall()

            with open("attendance.csv", "r") as f:
                lines = f.readlines()
                for student in students:
                    id, name, class_name, contact, address = student
                    count = 0
                    for line in lines:
                        if line.startswith(id):
                            count += 1
                    if count < 4:
                        status = "Cấm thi"
                    else:
                        status = "Được thi"
                    tk.Label(
                        self.window5,
                        text=f"{name} - {id} - {class_name} - {contact} - {address} - {count} - {status}",
                    ).pack()

            conn.close()
            back_button = tk.Button(
                self.window5, text="Quay lại", command=self.back_to_main5
            )
            back_button.pack()


if __name__ == "__main__":
    app = App()
    app.root.mainloop()
