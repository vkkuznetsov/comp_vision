import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import time


class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ЛАБ 3 КЗ Гордиенко Кузнецов ")

        self.image_path = None
        self.image = None
        self.tk_image = None

        self.create_menu()
        self.create_image_loading_interface()
        self.create_image_display_interface()
        self.create_sobel_interface()

        self.create_log_dog_methods_interface()

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

    def create_sobel_interface(self):
        self.sobel_filters = tk.LabelFrame(self.root, text="Фильтр Собеля")
        self.sobel_filters.pack(pady=10)

        self.three = tk.Button(self.sobel_filters, text="3х3",
                               command=lambda: self.apply_sobel_filter_manual(3))
        self.three.grid(row=0, column=0, padx=5)

        self.five = tk.Button(self.sobel_filters, text="5х5",
                              command=lambda: self.apply_sobel_filter_manual(5))
        self.five.grid(row=0, column=1, padx=5)

        self.seven = tk.Button(self.sobel_filters, text="7х7",
                               command=lambda: self.apply_sobel_filter_manual(7))
        self.seven.grid(row=0, column=2, padx=5)

        self.compare_button = tk.Button(self.sobel_filters, text="Сравнить фильтры",
                                        command=self.compare_filters)
        self.compare_button.grid(row=1, column=1, pady=5)

    def create_log_dog_methods_interface(self):
        self.log_dog = tk.LabelFrame(self.root, text="LoG и DoG методы детектирования границ")
        self.log_dog.pack(pady=10)

        self.log = tk.Button(self.log_dog, text="LoG-метод",
                             command=lambda: self.log_edge_detection(sigma=float(self.sigma_entry1.get())))
        self.log.grid(row=0, column=2, padx=5)

        self.dog = tk.Button(self.log_dog, text="DoG-метод", command=lambda: self.dog_edge_detection(
            sigma1=float(self.sigma1_entry.get()), sigma2=float(self.sigma2_entry.get())))
        self.dog.grid(row=2, column=2, padx=5)

        # Ввод для sigma LoG
        self.sigma_label1 = tk.Label(self.log_dog, text="Sigma для LoG:")
        self.sigma_label1.grid(row=0, column=0, padx=5, sticky='e')

        self.sigma_entry1 = tk.Entry(self.log_dog)
        self.sigma_entry1.grid(row=0, column=1, padx=5)
        self.sigma_entry1.insert(0, "1.0")

        # Ввод для sigma1 DoG
        self.sigma1_label = tk.Label(self.log_dog, text="Sigma1 для DoG:")
        self.sigma1_label.grid(row=2, column=0, padx=5, sticky='e')

        self.sigma1_entry = tk.Entry(self.log_dog)
        self.sigma1_entry.grid(row=2, column=1, padx=5)
        self.sigma1_entry.insert(0, "0.5")

        # Ввод для sigma2 DoG
        self.sigma2_label = tk.Label(self.log_dog, text="Sigma2 для DoG:")
        self.sigma2_label.grid(row=3, column=0, padx=5, sticky='e')

        self.sigma2_entry = tk.Entry(self.log_dog)
        self.sigma2_entry.grid(row=3, column=1, padx=5)
        self.sigma2_entry.insert(0, "1.0")

        self.sigma_label = tk.Label(self.log_dog, text="Sigmas для сравнения:")
        self.sigma_label.grid(row=1, column=0, padx=5, sticky='e')

        self.sigma_entry = tk.Entry(self.log_dog)
        self.sigma_entry.grid(row=1, column=1, pady=5)
        self.sigma_entry.insert(0, "0.5, 1, 3, 5, 7")

        self.compare_log_button = tk.Button(
            self.log_dog, text="Сравнить LoG", command=self.run_compare_log_sigmas
        )
        self.compare_log_button.grid(row=1, column=2, pady=5)

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

    def apply_sobel_filter_manual(self, size=3):
        if self.image is None:
            return

        start_time = time.time()

        image_array = np.array(self.image.convert('L'))

        if size == 3:
            sobel_kernel_x = np.array([[-1, 0, 1],
                                       [-2, 0, 2],
                                       [-1, 0, 1]])
            sobel_kernel_y = np.array([[-1, -2, -1],
                                       [0, 0, 0],
                                       [1, 2, 1]])
        elif size == 5:
            sobel_kernel_x = np.array([[-2, -1, 0, 1, 2],
                                       [-2, -1, 0, 1, 2],
                                       [-4, -2, 0, 2, 4],
                                       [-2, -1, 0, 1, 2],
                                       [-2, -1, 0, 1, 2]
                                       ])
            sobel_kernel_y = np.array([[-2, -2, -4, -2, -2],
                                       [-1, -1, -2, -1, -1],
                                       [0, 0, 0, 0, 0],
                                       [1, 1, 2, 1, 1],
                                       [2, 2, 4, 2, 2]
                                       ])
        elif size == 7:
            sobel_kernel_x = np.array([[-3, -2, -1, 0, 1, 2, 3],
                                       [-3, -2, -1, 0, 1, 2, 3],
                                       [-3, -2, -1, 0, 1, 2, 3],
                                       [-4, -3, -2, 0, 2, 3, 4],
                                       [-3, -2, -1, 0, 1, 2, 3],
                                       [-3, -2, -1, 0, 1, 2, 3],
                                       [-3, -2, -1, 0, 1, 2, 3]
                                       ])
            sobel_kernel_y = np.array([[-3, -3, -3, -4, -3, -3, -3],
                                       [-2, -2, -2, -3, -2, -2, -2],
                                       [-1, -1, -1, -2, -1, -1, -1],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [1, 1, 1, 2, 1, 1, 1],
                                       [2, 2, 2, 3, 2, 2, 2],
                                       [3, 3, 3, 4, 3, 3, 3]
                                       ])

        height, width = image_array.shape

        sobel_image = np.zeros((height, width))

        for y in range(size // 2, height - size // 2):
            for x in range(size // 2, width - size // 2):
                # Вычисляем горизонтальный и вертикальный градиенты
                gx = np.sum(np.multiply(sobel_kernel_x,
                                        image_array[y - size // 2:y + size // 2 + 1, x - size // 2:x + size // 2 + 1]))
                gy = np.sum(np.multiply(sobel_kernel_y,
                                        image_array[y - size // 2:y + size // 2 + 1, x - size // 2:x + size // 2 + 1]))

                # Вычисляем магнитуду градиента
                sobel_image[y, x] = np.sqrt(gx ** 2 + gy ** 2)

        sobel_image = np.clip(sobel_image / np.max(sobel_image) * 255, 0, 255).astype(np.uint8)

        execution_time = time.time() - start_time

        sobel_image_pil = Image.fromarray(sobel_image)

        self.tk_image = ImageTk.PhotoImage(image=sobel_image_pil)
        self.canvas.create_image(300, 300, image=self.tk_image)

        return sobel_image, execution_time

    def compare_filters(self):
        sizes = [3, 5, 7]
        fig, axes = plt.subplots(1, len(sizes), figsize=(15, 5))
        execution_times = []

        for i, size in enumerate(sizes):
            filtered_image, exec_time = self.apply_sobel_filter_manual(size)
            execution_times.append(exec_time)

            if filtered_image is not None:
                axes[i].imshow(filtered_image, cmap='gray')
                title = f"Фильтр {size}x{size}\nВремя: {exec_time:.4f} сек"
                axes[i].set_title(title)
                axes[i].axis('off')

        plt.show()

    def log_edge_detection(self, sigma=5.0, display=True):
        image = Image.open(self.image_path)
        image_array = np.array(image.convert('L'))

        blurred = self.gaussian_blur(sigma, image_array)

        laplacian_kernel = np.array([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]])
        height, width = image_array.shape
        log_image = np.zeros_like(image_array)

        for y in range(1, height - 1):
            for x in range(1, width - 1):
                # Извлекаем подматрицу 3x3 вокруг текущего пикселя
                region = blurred[y - 1:y + 2, x - 1:x + 2]

                # Применяем ядро Лапласа к подматрице
                filtered_value = np.sum(region * laplacian_kernel)

                # Сохраняем результат свертки
                log_image[y, x] = filtered_value

        log_image = np.clip(log_image, 0, 255)

        if not display:
            return Image.fromarray(np.uint8(log_image))

        self.display_image(Image.fromarray(np.uint8(log_image)))

    def compare_log_sigmas(self, sigmas):
        fig, axes = plt.subplots(1, len(sigmas), figsize=(20, 5))
        for i, sigma in enumerate(sigmas):

            log_image = self.log_edge_detection(sigma, display=False)

            ax = axes[i]
            ax.imshow(log_image, cmap='gray')
            ax.set_title(f"LoG with sigma={sigma}")
            ax.axis('off')

        plt.show()

    def run_compare_log_sigmas(self):
        sigma_values = self.sigma_entry.get()

        sigmas = list(map(float, sigma_values.split(',')))

        self.compare_log_sigmas(sigmas)

    def dog_edge_detection(self, sigma1=0.5, sigma2=5.0):
        image = Image.open(self.image_path)
        image_array = np.array(image.convert('L'))

        blur1 = self.gaussian_blur(sigma1, image_array)
        blur2 = self.gaussian_blur(sigma2, image_array)

        dog_image = blur1 - blur2

        dog_image = np.clip(dog_image, 0, 255)
        self.display_image(Image.fromarray(np.uint8(dog_image)))

    def gaussian_blur(self, sigma, image_array=None):
        if image_array is None:
            image = Image.open(self.image_path)
            image_array = np.array(image)

        kernel_size = 6 * sigma + 1
        kernel = self.gaussian_kernel(kernel_size, sigma)

        blurred_image_array = self.apply_convolution(image_array, kernel)

        return blurred_image_array

    def apply_convolution(self, image_array, kernel):
        if len(image_array.shape) == 2:
            output = convolve2d(image_array, kernel, mode='same', boundary='symm')
        return output

    def gaussian_kernel(self, size, sigma=1.0):
        x, y = np.meshgrid(np.arange(-size // 2 + 1, size // 2 + 1),
                           np.arange(-size // 2 + 1, size // 2 + 1))
        kernel = np.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))
        return kernel / np.sum(kernel)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
