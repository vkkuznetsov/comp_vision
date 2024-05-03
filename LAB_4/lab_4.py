import tkinter as tk
from tkinter import filedialog
import copy
import cv2
from PIL import Image, ImageTk
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import scipy.signal


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
        self.threshold_scale_k = tk.Scale(self.threshold_frame, from_=3, to=51, orient=tk.HORIZONTAL, label="Размер k", resolution=2)
        self.threshold_scale_k.grid(row=len(methods) + 2, column=2, columnspan=2, pady=5)

        self.threshold_button = tk.Button(self.threshold_frame, text="Адаптивный порог",
                                          command=self.adaptive_threashold)
        self.threshold_button.grid(row=len(methods) + 3, column=2, columnspan=2, pady=5)

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

    # пункт 2  P-tile
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
            smooth_levels = [0, 1, 3, 5]
            fig, axes = plt.subplots(len(smooth_levels), 2, figsize=(10, 15))
            for i, smooth in enumerate(smooth_levels):
                smoothed_image = self.smooth_image(grayscale_image, smooth)
                while True:
                    # Разделение пикселей на два класса (фон и объекты) по текущему порогу
                    foreground_pixels = smoothed_image[smoothed_image > threshold]
                    background_pixels = smoothed_image[smoothed_image <= threshold]

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
                axes[i, 0].imshow(segmented_image, cmap='gray')
                axes[i, 0].set_title(f'Сглаживание = {smooth})')
                axes[i, 0].axis('off')
                self.compare_smoothing(grayscale_image, axes[i, -1], smooth)
            plt.tight_layout()
            plt.show()

    # Метод к-средних

    def k_means_segmentation(self):
        if self.image is not None:
            grayscale_image = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2GRAY)

            k_values = [2, 3, 5, 10]
            smooth_levels = [0, 1, 3, 5]
            fig, axes = plt.subplots(len(k_values), len(smooth_levels) + 1, figsize=(25, 17))

            for i, smooth in enumerate(smooth_levels):
                smoothed_image = self.smooth_image(grayscale_image, smooth)
                for j, k in enumerate(k_values):
                    kmeans = KMeans(n_clusters=k, random_state=0).fit(smoothed_image.reshape(-1, 1))
                    labels = kmeans.labels_
                    centers = kmeans.cluster_centers_

                    # Вычисление среднеквадратичного отклонения от центроидов
                    score = np.mean((smoothed_image.reshape(-1, 1) - centers[labels]) ** 2)

                    # Бинаризация изображения по лучшему порогу
                    binary_image = np.where(smoothed_image > np.mean(centers), 255, 0).astype(np.uint8)

                    # Отображение сегментированного изображения
                    segmented_image = Image.fromarray(binary_image)

                    axes[i, j].imshow(segmented_image, cmap='gray')
                    axes[i, j].set_title(f'(k={k}, smooth={smooth}), Отклонение: {score:.2f}')
                    axes[i, j].axis('off')

                    # Вызов метода для отображения гистограммы
                self.compare_smoothing(grayscale_image, axes[i, -1], smooth_levels[i])

            plt.tight_layout()
            plt.show()

    def smooth_image(self, image, smooth_level):
        # Применяем сглаживание к изображению
        smoothed_image = cv2.blur(image, (smooth_level * 2 + 1, smooth_level * 2 + 1))
        return smoothed_image

    def compare_smoothing(self, image, ax, smooth):
        histogram, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))

        smoothed_hist = self.smooth_histogram(histogram, iterations=smooth)
        peaks, _ = scipy.signal.find_peaks(smoothed_hist)

        ax.plot(smoothed_hist)
        ax.plot(peaks, smoothed_hist[peaks], "x")
        ax.set_title(f'После {smooth} сглаживаний')
        ax.set_xlabel('Интенсивность пикселя')
        ax.set_ylabel('Частота')

    def smooth_histogram(self, histogram, iterations):
        for _ in range(iterations):
            histogram = np.convolve(histogram, [1 / 3, 1 / 3, 1 / 3], mode='same')
        return histogram

    # Пункт 3 адаптивный порог
    def adaptive_threashold(self):
        if self.image is not None:
            grayscale_image = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2GRAY)

            # Получение выбранного метода
            method = self.method_var.get()

            # Получение выбранного значения порога
            threshold_value = self.threshold_scale.get()
            k = int(self.threshold_scale_k.get())
            # Применение выбранного метода порогования
            if method == "Среднее":
                thresholded_image = cv2.adaptiveThreshold(grayscale_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                          cv2.THRESH_BINARY, k, threshold_value)
            elif method == "Медиана":
                thresholded_image = cv2.adaptiveThreshold(grayscale_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                          cv2.THRESH_BINARY, k, threshold_value)
            elif method == "(min+max)/2":
                thresholded_image = cv2.adaptiveThreshold(grayscale_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                          cv2.THRESH_BINARY, k, threshold_value)
                thresholded_image += cv2.adaptiveThreshold(grayscale_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                           cv2.THRESH_BINARY_INV, k, threshold_value)
                thresholded_image //= 2  # Применение формулы (min+max)/2

            # Отображение сегментированного изображения
            segmented_image = Image.fromarray(thresholded_image)
            self.display_image(segmented_image)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
