import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from scipy.signal import convolve2d


class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing App")

        self.image_path = None
        self.image = None
        self.tk_image = None  # Для хранения отображаемого изображения

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

    def create_log_dog_methods_interface(self):
        self.log_dog = tk.LabelFrame(self.root, text="LoG и DoG методы детектирования границ")
        self.log_dog.pack(pady=10)

        self.log = tk.Button(self.log_dog, text="LoG-метод", command=self.log_edge_detection)
        self.log.grid(row=0, column=1, padx=5)

        self.dog = tk.Button(self.log_dog, text="DoG-метод", command=self.dog_edge_detection)
        self.dog.grid(row=0, column=2, padx=5)

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

        # Преобразуем PIL-изображение в массив numpy в оттенках серого
        image_array = np.array(self.image.convert('L'))

        # Определяем ядра фильтра Собеля для горизонтального и вертикального градиента
        if size == 3:
            sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        elif size == 5:
            sobel_kernel_x = np.array([[-2, -1, 0, 1, 2],
                                       [-2, -1, 0, 1, 2],
                                       [-2, -1, 0, 1, 2],
                                       [-2, -1, 0, 1, 2],
                                       [-2, -1, 0, 1, 2]])
            sobel_kernel_y = np.array([[-2, -2, -2, -2, -2],
                                       [-1, -1, -1, -1, -1],
                                       [0, 0, 0, 0, 0],
                                       [1, 1, 1, 1, 1],
                                       [2, 2, 2, 2, 2]])
        elif size == 7:
            sobel_kernel_x = np.array([[-3, -2, -1, 0, 1, 2, 3],
                                       [-3, -2, -1, 0, 1, 2, 3],
                                       [-3, -2, -1, 0, 1, 2, 3],
                                       [-3, -2, -1, 0, 1, 2, 3],
                                       [-3, -2, -1, 0, 1, 2, 3],
                                       [-3, -2, -1, 0, 1, 2, 3],
                                       [-3, -2, -1, 0, 1, 2, 3]])
            sobel_kernel_y = np.array([[-3, -3, -3, -3, -3, -3, -3],
                                       [-2, -2, -2, -2, -2, -2, -2],
                                       [-1, -1, -1, -1, -1, -1, -1],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [1, 1, 1, 1, 1, 1, 1],
                                       [2, 2, 2, 2, 2, 2, 2],
                                       [3, 3, 3, 3, 3, 3, 3]])

        # Получаем размеры изображения
        height, width = image_array.shape

        # Создаем пустой массив для хранения результата
        sobel_image = np.zeros((height, width))

        # Применяем фильтр Собеля, игнорируя крайние пиксели
        for y in range(size // 2, height - size // 2):
            for x in range(size // 2, width - size // 2):
                # Вычисляем горизонтальный и вертикальный градиенты
                gx = np.sum(np.multiply(sobel_kernel_x,
                                        image_array[y - size // 2:y + size // 2 + 1, x - size // 2:x + size // 2 + 1]))
                gy = np.sum(np.multiply(sobel_kernel_y,
                                        image_array[y - size // 2:y + size // 2 + 1, x - size // 2:x + size // 2 + 1]))

                # Вычисляем магнитуду градиента
                sobel_image[y, x] = np.sqrt(gx ** 2 + gy ** 2)

        # Нормализуем изображение
        sobel_image = np.clip(sobel_image / np.max(sobel_image) * 255, 0, 255).astype(np.uint8)

        # Преобразуем обратно в PIL-изображение
        sobel_image_pil = Image.fromarray(sobel_image)

        # Обновляем изображение для отображения
        self.tk_image = ImageTk.PhotoImage(image=sobel_image_pil)
        self.canvas.create_image(300, 300, image=self.tk_image)

    def log_edge_detection(self, sigma=1.0):
        image = Image.open(self.image_path)
        image_array = np.array(image.convert('L'))  # Преобразование в оттенки серого

        # Гауссово размытие
        blurred = self.gaussian_blur(sigma, image_array)

        # Применение оператора Лапласа
        laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        log_image = convolve2d(blurred, laplacian_kernel, mode='same', boundary='symm')

        # Нормализация и отображение
        log_image = np.clip(log_image, 0, 255)
        self.display_image(Image.fromarray(np.uint8(log_image)))

    def dog_edge_detection(self, sigma1=1.0, sigma2=2.0):
        image = Image.open(self.image_path)
        image_array = np.array(image.convert('L'))  # Преобразование в оттенки серого

        # Два Гауссовых размытия
        blur1 = self.gaussian_blur(sigma1, image_array)
        blur2 = self.gaussian_blur(sigma2, image_array)

        # Разница Гауссиан
        dog_image = blur1 - blur2

        # Нормализация и отображение
        dog_image = np.clip(dog_image, 0, 255)
        self.display_image(Image.fromarray(np.uint8(dog_image)))

    def gaussian_blur(self, sigma, image_array=None):
        if image_array is None:
            image = Image.open(self.image_path)
            image_array = np.array(image)

        kernel_size = 6 * sigma + 1
        kernel = self.gaussian_kernel(kernel_size, sigma)

        blurred_image_array = self.apply_convolution(image_array, kernel)

        blurred_image = Image.fromarray(np.uint8(blurred_image_array))

        return blurred_image_array

    def apply_convolution(self, image_array, kernel):
        if len(image_array.shape) == 2:  # Если изображение в оттенках серого
            output = convolve2d(image_array, kernel, mode='same', boundary='symm')
        else:  # Если изображение цветное
            output = np.zeros_like(image_array)
            for c in range(image_array.shape[2]):  # Применяем свертку для каждого канала
                output[:, :, c] = convolve2d(image_array[:, :, c], kernel, mode='same', boundary='symm')
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
