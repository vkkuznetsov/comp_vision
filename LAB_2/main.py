import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt


class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing App")

        # Путь к текущему изображению
        self.image_path = None

        # Создание главного меню
        self.create_menu()

        # Создание интерфейса для загрузки изображения
        self.create_image_loading_interface()

        # Создание интерфейса для отображения изображения
        self.create_image_display_interface()

        # Создание интерфейса для цветности
        self.create_colorfulness_interface()

        # Создание интерфейса для сглаживания
        self.create_smoothing_interface()

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
        self.canvas = tk.Canvas(self.root, width=800, height=800)
        self.canvas.pack()

    def create_colorfulness_interface(self):
        self.colorfulness_frame = tk.LabelFrame(self.root, text="Цветность")
        self.colorfulness_frame.pack(pady=10)

        self.logarithmic_button = tk.Button(self.colorfulness_frame, text="Логарифмическое преобразование",
                                            command=self.logarithmic_transform)
        self.logarithmic_button.grid(row=0, column=0, padx=5)

        self.power_button = tk.Button(self.colorfulness_frame, text="Степенное преобразование",
                                      command=self.power_transform)
        self.power_button.grid(row=0, column=1, padx=5)

        self.binary_button = tk.Button(self.colorfulness_frame, text="Бинарное преобразование",
                                       command=self.binary_transform)
        self.binary_button.grid(row=0, column=2, padx=5)

        self.clip_button = tk.Button(self.colorfulness_frame, text="Вырезание диапозона", command=self.clip_range)
        self.clip_button.grid(row=0, column=3, padx=5)

    def create_smoothing_interface(self):
        self.smoothing_frame = tk.LabelFrame(self.root, text="Сглаживание")
        self.smoothing_frame.pack(pady=10)

        self.rectangular3_button = tk.Button(self.smoothing_frame, text="Прямоугольный фильтр (3x3)",
                                             command=lambda: self.apply_filter(3, 'rectangular'))
        self.rectangular3_button.grid(row=0, column=0, padx=5)

        self.rectangular5_button = tk.Button(self.smoothing_frame, text="Прямоугольный фильтр (5x5)",
                                             command=lambda: self.apply_filter(5, 'rectangular'))
        self.rectangular5_button.grid(row=0, column=1, padx=5)

        self.median3_button = tk.Button(self.smoothing_frame, text="Медианный фильтр (3x3)",
                                        command=lambda: self.apply_filter(3, 'median'))
        self.median3_button.grid(row=0, column=2, padx=5)

        self.median5_button = tk.Button(self.smoothing_frame, text="Медианный фильтр (5x5)",
                                        command=lambda: self.apply_filter(5, 'median'))
        self.median5_button.grid(row=0, column=3, padx=5)

        self.gaus_button = tk.Button(self.smoothing_frame, text='Гаусовское сглаживание',
                                     command=lambda: self.display_blur_images('gaus'))
        self.gaus_button.grid(row=1, column=0)

        self.sigma_button = tk.Button(self.smoothing_frame, text='Сигма фильтр',
                                      command=lambda: self.display_blur_images('sigma'))
        self.sigma_button.grid(row=1, column=1)

    def sigma_filter(self, sigma=1):
        """Применение сигма-фильтра к изображению"""
        image = Image.open(self.image_path)
        image_array = np.array(image)

        # Применение фильтра к каждому пикселю
        filtered_image_array = np.zeros_like(image_array, dtype=np.float32)
        height, width, channels = image_array.shape

        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    # Определение границ окна для текущего пикселя
                    y_min = max(0, y - sigma)
                    y_max = min(height - 1, y + sigma)
                    x_min = max(0, x - sigma)
                    x_max = min(width - 1, x + sigma)

                    # Вычисление среднего значения в окне
                    window = image_array[y_min:y_max + 1, x_min:x_max + 1, c]
                    average_value = np.mean(window)

                    # Замена пикселя средним значением окна
                    filtered_image_array[y, x, c] = average_value

        # Преобразование массива обратно в изображение
        filtered_image = Image.fromarray(np.uint8(filtered_image_array))
        return filtered_image

    def gaussian_blur(self, sigma=5.0):
        """Применение гауссовского сглаживания к изображению"""
        image = Image.open(self.image_path)
        image_array = np.array(image)

        # Создание ядра Гаусса
        kernel_size = int(6 * sigma + 1)  # Размер ядра рассчитывается на основе sigma
        kernel = self.gaussian_kernel(kernel_size, sigma)

        # Применение свертки с ядром Гаусса
        blurred_image_array = self.apply_convolution(image_array, kernel)

        # Преобразование массива обратно в изображение
        blurred_image = Image.fromarray(np.uint8(blurred_image_array))

        return blurred_image

    def apply_convolution(self, image_array, kernel):
        """Применение свертки к изображению с использованием заданного ядра"""
        output = np.zeros_like(image_array)
        for c in range(image_array.shape[2]):  # Проходим по каждому каналу цвета
            output[:, :, c] = convolve2d(image_array[:, :, c], kernel, mode='same', boundary='symm')
        return output

    def gaussian_kernel(self, size, sigma=1.0):
        """Функция для создания ядра Гаусса"""
        x, y = np.meshgrid(np.arange(-size // 2 + 1, size // 2 + 1),
                           np.arange(-size // 2 + 1, size // 2 + 1))
        kernel = np.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))
        return kernel / np.sum(kernel)

    def display_blur_images(self, mode):
        """Отображение изображений сглаженной версии с разными значениями сигмы"""
        sigmas = [1, 3, 10]

        # Создание нового окна графика
        fig, axs = plt.subplots(2, len(sigmas), figsize=(15, 10))

        for i, sigma in enumerate(sigmas):
            # Применение гауссовского сглаживания или сигма-фильтра
            if mode == 'gaus':
                original_image = Image.open(self.image_path)
                blurred_image = self.gaussian_blur(sigma)
            elif mode == 'sigma':
                original_image = Image.open(self.image_path)
                blurred_image = self.sigma_filter(sigma)

            # Отображение сглаженного изображения в подграфике
            axs[0, i].imshow(original_image)
            axs[0, i].set_title(f"Original (Sigma = {sigma})")
            axs[0, i].axis('off')

            axs[1, i].imshow(blurred_image)
            axs[1, i].set_title(f"Blurred (Sigma = {sigma})")
            axs[1, i].axis('off')

            # Вычисление абсолютной разности между исходным и сглаженным изображениями
            diff_array = np.abs(np.array(original_image) - np.array(blurred_image))

            # Отображение карты разности под каждым сглаженным изображением
            axs[1, i].imshow(diff_array)
            axs[1, i].set_title(f"Difference (Sigma = {sigma})")
            axs[1, i].axis('off')

        # Отображение окна графика с изображениями и картами разности
        plt.tight_layout()
        plt.show()

    def open_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")])
        if self.image_path:
            self.display_image()

    def display_image(self):
        image = Image.open(self.image_path)
        image.thumbnail((800, 800))  # Масштабирование изображения для отображения на холсте
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def apply_filter(self, kernel_size, filter_type):
        if self.image_path:
            image = Image.open(self.image_path)
            if filter_type == 'rectangular':
                transformed_image = self.apply_rectangular_filter(image, kernel_size)
            elif filter_type == 'median':
                transformed_image = self.apply_median_filter(image, kernel_size)
            self.display_transformed_image(transformed_image)

    def apply_rectangular_filter(self, image, kernel_size):
        # Применение прямоугольного фильтра
        filtered_image = cv2.blur(np.array(image), (kernel_size, kernel_size))
        return Image.fromarray(filtered_image)

    def apply_median_filter(self, image, kernel_size):
        # Применение медианного фильтра
        filtered_image = cv2.medianBlur(np.array(image), kernel_size)
        return Image.fromarray(filtered_image)

    def logarithmic_transform(self):
        if self.image_path:
            image = Image.open(self.image_path)
            image_array = np.array(image)
            if len(image_array[0][0]) > 3:
                image_array = np.delete(image_array, 3, axis=2)
            c = 255 / np.log(1 + np.max(image_array))
            log_transformed = c * np.log(1 + image_array)
            log_transformed = np.uint8(log_transformed)
            log_image = Image.fromarray(log_transformed)
            self.display_transformed_image(log_image)

    def power_transform(self):
        if self.image_path:
            gamma = 1.5  # Произвольное значение гаммы
            c = 1  # Произвольное значение коэффициента
            image = Image.open(self.image_path)
            image_array = np.array(image)
            power_transformed = c * np.power(image_array, gamma)
            power_transformed = np.uint8(power_transformed)
            power_image = Image.fromarray(power_transformed)
            self.display_transformed_image(power_image)

    def binary_transform(self):
        if self.image_path:
            threshold = 128  # Произвольное пороговое значение
            image = Image.open(self.image_path)
            image_array = np.array(image)
            binary_transformed = np.where(image_array < threshold, 0, 255)
            binary_image = Image.fromarray(np.uint8(binary_transformed))
            self.display_transformed_image(binary_image)

    def clip_range(self):
        if self.image_path:
            min_val = 50  # Произвольное минимальное значение диапазона
            max_val = 200  # Произвольное максимальное значение диапазона
            constant_value = 100  # Произвольное константное значение для обработки пикселей вне диапазона
            image = Image.open(self.image_path)
            image_array = np.array(image)
            clipped_image = np.clip(image_array, min_val, max_val)
            clipped_image[np.where((image_array < min_val) | (image_array > max_val))] = constant_value
            clipped_image = np.uint8(clipped_image)
            clipped_image = Image.fromarray(clipped_image)
            self.display_transformed_image(clipped_image)

    def display_transformed_image(self, image):
        image.thumbnail((800, 800))
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
