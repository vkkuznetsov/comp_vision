import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np


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
