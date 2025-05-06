import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# 忽略 TensorFlow 下載模型過程中的錯誤顯示（可選）
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 取得特徵模型
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array, verbose=0)
    return features

def load_database_features(database_path):
    features = []
    labels = []
    for label in os.listdir(database_path):
        class_folder = os.path.join(database_path, label)
        if not os.path.isdir(class_folder):
            continue
        for fname in os.listdir(class_folder):
            img_path = os.path.join(class_folder, fname)
            try:
                feat = extract_features(img_path)
                features.append(feat[0])
                labels.append(label)
            except Exception as e:
                print(f"無法讀取 {img_path}: {e}")
    return np.array(features), labels

def find_match(img_path, database_features, database_labels, threshold=0.8):
    query_feat = extract_features(img_path)
    sims = cosine_similarity(query_feat, database_features)[0]
    max_idx = np.argmax(sims)
    max_sim = sims[max_idx]
    if max_sim > threshold:
        return "Pass", database_labels[max_idx], max_sim
    else:
        return "NG", database_labels[max_idx], max_sim

# GUI 介面
class ImageComparerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("影像比對辨識系統")

        self.database_path = tk.StringVar()
        self.img_path = tk.StringVar()

        tk.Label(root, text="資料庫資料夾:").pack()
        tk.Entry(root, textvariable=self.database_path, width=50).pack()
        tk.Button(root, text="選擇資料夾", command=self.select_database_folder).pack()

        tk.Label(root, text="選擇比對圖片:").pack()
        tk.Entry(root, textvariable=self.img_path, width=50).pack()
        tk.Button(root, text="選擇圖片", command=self.select_image).pack()

        tk.Button(root, text="開始比對", command=self.compare).pack(pady=10)
        self.result_label = tk.Label(root, text="結果：")
        self.result_label.pack()

        self.canvas = tk.Canvas(root, width=224, height=224)
        self.canvas.pack()

        self.db_features = None
        self.db_labels = None

    def select_database_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.database_path.set(path)
            self.db_features, self.db_labels = load_database_features(path)
            messagebox.showinfo("完成", "資料庫特徵載入完畢")

    def select_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if path:
            self.img_path.set(path)
            img = Image.open(path).resize((224, 224))
            self.tkimg = ImageTk.PhotoImage(img)
            self.canvas.create_image(112, 112, image=self.tkimg)

    def compare(self):
        if self.db_features is None or not self.img_path.get():
            messagebox.showerror("錯誤", "請確認資料庫與圖片皆已選擇")
            return
        result, match_label, similarity = find_match(
            self.img_path.get(), self.db_features, self.db_labels
        )
        self.result_label.config(text=f"結果：{result}\n最相似類別：{match_label} (相似度：{similarity:.2f})")

if __name__ == '__main__':
    root = tk.Tk()
    app = ImageComparerApp(root)
    root.mainloop()