import tkinter as tk
from tkinter import filedialog

import cv2
from PIL import Image, ImageTk
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


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
        self.create_threshold_interface1()
        self.create_threshold_interface2()
        self.create_threshold_interface3()

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
        self.load_image_button.grid(row=0, column=1, pady=10)

    def create_image_display_interface(self):
        self.canvas = tk.Canvas(self.root, width=600, height=600)
        self.canvas.grid(row=1, column=0, columnspan=3)

    def create_threshold_interface1(self):
        self.threshold_frame3 = tk.LabelFrame(self.root, text="ПУНКТ 1.")
        self.threshold_frame3.grid(row=2, column=0, padx=20, pady=20)

        self.nearing = tk.Button(self.threshold_frame3, text="Алгоритм выделения краев",
                                 command=self.segment_by_edge_detection)
        self.nearing.pack()

    def create_threshold_interface2(self):
        self.threshold_frame2 = tk.LabelFrame(self.root, text="ПУНКТ 2.")
        self.threshold_frame2.grid(row=2, column=1, padx=10, pady=10)

        self.threshold_scale2 = tk.Scale(self.threshold_frame2, from_=0, to=100, orient=tk.HORIZONTAL,
                                         label="Процентиль")
        self.threshold_scale2.set(95)
        self.threshold_scale2.pack()

        self.segment_button = tk.Button(self.threshold_frame2, text="P-tile",
                                        command=self.p_tile_segmentation)
        self.segment_button.pack()

        self.edging = tk.Button(self.threshold_frame2, text="Последовательное приближение",
                                command=self.consistent_nearing_segmentation)
        self.edging.pack(pady=5)
        self.kmeans = tk.Button(self.threshold_frame2, text="К-means", command=self.k_means_segmentation)
        self.kmeans.pack(pady=5)

    def create_threshold_interface3(self):
        self.threshold_frame = tk.LabelFrame(self.root, text="ПУНКТ 3")
        self.threshold_frame.grid(row=2, column=2, padx=10, pady=10)



        self.method_var = tk.StringVar()
        self.method_var.set("P-tile")

        methods = ["Среднее", "Медиана", "(min+max)/2"]
        for i, method in enumerate(methods):
            tk.Radiobutton(self.threshold_frame, text=method, variable=self.method_var, value=method).grid(row=i + 1,
                                                                                                           column=2,
                                                                                                           sticky="w",
                                                                                                           padx=5)

        self.threshold_scale = tk.Scale(self.threshold_frame, from_=0, to=255, orient=tk.HORIZONTAL, label="Порог")
        self.threshold_scale.grid(row=len(methods) + 1, column=2, columnspan=2, pady=5)

        self.threshold_button = tk.Button(self.threshold_frame, text="Адаптивный порог", command=self.apply_threshold)
        self.threshold_button.grid(row=len(methods) + 2, column=2, columnspan=2, pady=5)

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

    # пункт 1
    def segment_by_edge_detection(self):
        # Преобразование изображения в оттенки серого
        grayscale_image = np.array(self.image.convert("L"))

        # Применение алгоритма выделения краев
        edges = cv2.Canny(grayscale_image, 100, 200)  # Пример использования оператора Кэнни

        # Бинаризация краевого изображения
        binary_image = np.where(edges > 0, 255, 0).astype(np.uint8)

        segmented_image = Image.fromarray(binary_image)
        self.display_image(segmented_image)

    # пункт P-tile
    def p_tile_segmentation(self):
        if self.image is not None:
            # Преобразование изображения в оттенки серого
            grayscale_image = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2GRAY)

            # Вычисление порога на основе процентиля (P-tile)
            percentile = self.threshold_scale2.get()
            threshold = np.percentile(grayscale_image, percentile)

            # Бинаризация изображения по порогу
            binary_image = np.where(grayscale_image >= threshold, 255, 0).astype(np.uint8)

            # Отображение сегментированного изображения
            segmented_image = Image.fromarray(binary_image)
            self.display_image(segmented_image)

    # Последовательные приближения
    def consistent_nearing_segmentation(self):
        if self.image is not None:
            grayscale_image = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2GRAY)

            percentile = self.threshold_scale2.get()
            threshold = np.percentile(grayscale_image, percentile)

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

    # Метод к-средних
    def k_means_segmentation(self):
        if self.image is not None:
            grayscale_image = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2GRAY)

            k_values = [2, 3, 4, 5]

            fig, axes = plt.subplots(1, len(k_values), figsize=(16, 6))

            for i, k in enumerate(k_values):
                # Применение метода k-средних
                kmeans = KMeans(n_clusters=k, random_state=0).fit(grayscale_image.reshape(-1, 1))
                labels = kmeans.labels_
                centers = kmeans.cluster_centers_

                # Вычисление среднеквадратичного отклонения от центроидов
                score = np.mean((grayscale_image.reshape(-1, 1) - centers[labels]) ** 2)

                # Бинаризация изображения по лучшему порогу
                binary_image = np.where(grayscale_image > np.mean(centers), 255, 0).astype(np.uint8)

                # Отображение сегментированного изображения
                segmented_image = Image.fromarray(binary_image)

                # Отображение на одном холсте
                axes[i].imshow(segmented_image, cmap='gray')
                axes[i].set_title(f'(k={k}), Отклонение: {score:.2f}')
                axes[i].axis('off')

            plt.tight_layout()
            plt.show()

    def apply_threshold(self):
        if self.image is not None:
            method = self.method_var.get()
            threshold = self.threshold_scale.get()

            grayscale_image = self.image.convert("L")
            grayscale_array = np.array(grayscale_image)

            if method == "P-tile":
                histogram = calculate_and_plot_histogram(grayscale_array)
                threshold = ptile_threshold(histogram, threshold)
            elif method == "Mean":
                threshold = np.mean(grayscale_array)
            elif method == "Median":
                threshold = np.median(grayscale_array)

            binarized_image = binarize_image(grayscale_array, threshold)
            binarized_image = Image.fromarray(binarized_image)
            self.display_image(binarized_image)


def calculate_and_plot_histogram(image):
    # Построение гистограммы
    compare_smoothing(image)
    plt.figure()
    histogram, bin_edges, _ = plt.hist(image.flatten(), bins=256, range=(0, 256), color='gray', edgecolor="black")
    plt.title('Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.show()

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


import scipy.signal


def compare_smoothing(image, smooth_levels=[1, 3, 5, 10, 15]):
    histogram, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))

    for level in smooth_levels:
        smoothed_hist = smooth_histogram(histogram, iterations=level)
        peaks, _ = scipy.signal.find_peaks(smoothed_hist)

        plt.figure()
        plt.plot(smoothed_hist)
        plt.plot(peaks, smoothed_hist[peaks], "x")
        plt.title(f'Smoothed Histogram with {level} iterations')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.show()


def smooth_histogram(histogram, iterations=3):
    for _ in range(iterations):
        histogram = np.convolve(histogram, [1 / 3, 1 / 3, 1 / 3], mode='same')
    return histogram


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
