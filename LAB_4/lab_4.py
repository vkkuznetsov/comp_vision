import tkinter as tk
from tkinter import filedialog

import cv2
from PIL import Image, ImageTk
import numpy as np
from sklearn.cluster import KMeans


class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ЛАБ 4")

        self.image_path = None
        self.image = None
        self.tk_image = None

        self.create_menu()
        self.create_image_loading_interface()
        self.create_image_display_interface()
        self.create_threshold_interface()
        self.create_threshold_interface2()

    def create_menu(self):
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Открыть изображение", command=self.open_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)

    def create_image_loading_interface(self):
        self.load_image_button = tk.Button(self.root, text="Загрузить изображение", command=self.open_image)
        self.load_image_button.pack(pady=10)

    def create_image_display_interface(self):
        self.canvas = tk.Canvas(self.root, width=600, height=600)
        self.canvas.pack()

    def create_threshold_interface(self):
        self.threshold_frame = tk.LabelFrame(self.root, text="Выбор порога")
        self.threshold_frame.pack(pady=10)

        self.edging = tk.Button(self.threshold_frame, text="Выделение краев", command=self.apply_segmentation)
        self.edging.pack()

        self.method_var = tk.StringVar()
        self.method_var.set("P-tile")

        methods = ["P-tile", "Mean", "Median"]
        for method in methods:
            tk.Radiobutton(self.threshold_frame, text=method, variable=self.method_var, value=method).pack(anchor=tk.W)

        self.threshold_scale = tk.Scale(self.threshold_frame, from_=0, to=255, orient=tk.HORIZONTAL, label="Порог")
        self.threshold_scale.pack()

        self.threshold_button = tk.Button(self.threshold_frame, text="Применить порог", command=self.apply_threshold)
        self.threshold_button.pack()

        self.approximation = tk.Button(self.threshold_frame, text="Последовательное приближение", command=self.apply_segmentation)
        self.approximation.pack()

        self.kmeans = tk.Button(self.threshold_frame, text="K-Средних",
                                       command=self.apply_k_means())
        self.kmeans.pack()

    def apply_segmentation(self):
        if self.image is not None:
            grayscale_image = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2GRAY)

            # Инициализация начального порога
            threshold = 128

            # Максимальное количество итераций
            max_iterations = 100
            iteration = 0

            while True:
                # Разделение пикселей на два класса (фон и объекты) по текущему порогу
                foreground_pixels = grayscale_image[grayscale_image > threshold]
                background_pixels = grayscale_image[grayscale_image <= threshold]

                # Пересчет средних значений яркости для двух классов
                foreground_mean = np.mean(foreground_pixels)
                background_mean = np.mean(background_pixels)

                # Обновление порога
                new_threshold = (foreground_mean + background_mean) / 2

                # Проверка на сходимость
                if abs(new_threshold - threshold) < 1 or iteration >= max_iterations:
                    break

                threshold = new_threshold
                iteration += 1

            # Бинаризация изображения по полученному порогу
            binary_image = np.where(grayscale_image > threshold, 255, 0).astype(np.uint8)

            # Отображение сегментированного изображения
            segmented_image = Image.fromarray(binary_image)
            self.display_image(segmented_image)

    from sklearn.cluster import KMeans

    def apply_k_means(self):
        if self.image is not None:
            grayscale_image = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2GRAY)

            # Инициализация параметров k-средних
            k_values = [2, 3, 4, 5]  # Различные значения k для сравнения
            best_threshold = None
            best_score = float('inf')

            for k in k_values:
                # Применение метода k-средних
                kmeans = KMeans(n_clusters=k, random_state=0).fit(grayscale_image.reshape(-1, 1))
                labels = kmeans.labels_
                centers = kmeans.cluster_centers_

                # Вычисление среднеквадратичного отклонения от центроидов
                score = np.mean((grayscale_image.reshape(-1, 1) - centers[labels]) ** 2)

                # Выбор наилучшего порога на основе минимального среднеквадратичного отклонения
                if score < best_score:
                    best_score = score
                    best_threshold = np.mean(centers)

            # Бинаризация изображения по лучшему порогу
            binary_image = np.where(grayscale_image > best_threshold, 255, 0).astype(np.uint8)

            # Отображение сегментированного изображения
            segmented_image = Image.fromarray(binary_image)
            self.display_image(segmented_image)

    def create_threshold_interface2(self):
        self.threshold_frame = tk.LabelFrame(self.root, text="Выбор порога")
        self.threshold_frame.pack(pady=10)

        self.threshold_scale = tk.Scale(self.threshold_frame, from_=0, to=100, orient=tk.HORIZONTAL, label="Процентиль")
        self.threshold_scale.set(95)
        self.threshold_scale.pack()

        self.segment_button = tk.Button(self.root, text="Применить сегментацию", command=self.apply_segmentation2)
        self.segment_button.pack()

    def open_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")])
        if self.image_path:
            self.image = Image.open(self.image_path)
            self.display_image()

    def display_image(self, image=None):
        if image is None and self.image_path:
            image = Image.open(self.image_path)
        if image:
            image = image.resize((600, 600), Image.Resampling.LANCZOS)
            self.tk_image = ImageTk.PhotoImage(image)
            self.canvas.create_image(300, 300, image=self.tk_image)

    def apply_threshold(self):
        if self.image is not None:
            method = self.method_var.get()
            threshold = self.threshold_scale.get()

            grayscale_image = self.image.convert("L")
            grayscale_array = np.array(grayscale_image)

            if method == "P-tile":
                histogram = calculate_histogram(grayscale_array)
                threshold = ptile_threshold(histogram, threshold)
            elif method == "Mean":
                threshold = np.mean(grayscale_array)
            elif method == "Median":
                threshold = np.median(grayscale_array)

            binarized_image = binarize_image(grayscale_array, threshold)
            binarized_image = Image.fromarray(binarized_image)
            self.display_image(binarized_image)

    def apply_segmentation(self):
        if self.image is not None:
            grayscale_image = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2GRAY)

            # Applying Sobel edge detection
            sobel_x = cv2.Sobel(grayscale_image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(grayscale_image, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

            # Binarizing the edge image
            threshold = 50  # You can adjust this threshold as needed
            binary_image = np.where(magnitude >= threshold, 255, 0).astype(np.uint8)

            # Displaying the segmented image
            segmented_image = Image.fromarray(binary_image)
            self.display_image(segmented_image)
    def apply_segmentation2(self):
        if self.image is not None:
            # Преобразование изображения в оттенки серого
            grayscale_image = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2GRAY)

            # Вычисление порога на основе процентиля (P-tile)
            percentile = self.threshold_scale.get()
            threshold = np.percentile(grayscale_image, percentile)

            # Бинаризация изображения по порогу
            binary_image = np.where(grayscale_image >= threshold, 255, 0).astype(np.uint8)

            # Отображение сегментированного изображения
            segmented_image = Image.fromarray(binary_image)
            self.display_image(segmented_image)

def calculate_histogram(image):
    histogram, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
    return histogram


def ptile_threshold(histogram, percentile):
    total_pixels = np.sum(histogram)
    target_pixels = total_pixels * (percentile / 100)
    cumulative_sum = 0
    threshold = 0
    for i in range(len(histogram)):
        cumulative_sum += histogram[i]
        if cumulative_sum >= target_pixels:
            threshold = i
            break
    return threshold


def binarize_image(image, threshold):
    binarized_image = np.where(image >= threshold, 255, 0)
    return binarized_image.astype(np.uint8)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
