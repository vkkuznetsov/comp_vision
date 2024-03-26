import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np


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
        self.create_colorfulness_interface()

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

    def create_colorfulness_interface(self):
        self.colorfulness_frame = tk.LabelFrame(self.root, text="Фильтр Собеля")
        self.colorfulness_frame.pack(pady=10)

        self.logarithmic_button = tk.Button(self.colorfulness_frame, text="3х3",
                                            command=lambda: self.apply_sobel_filter_manual(3))
        self.logarithmic_button.grid(row=0, column=0, padx=5)

        self.power_button = tk.Button(self.colorfulness_frame, text="5х5",
                                      command=lambda: self.apply_sobel_filter_manual(5))
        self.power_button.grid(row=0, column=1, padx=5)

        self.binary_button = tk.Button(self.colorfulness_frame, text="7х7",
                                       command=lambda: self.apply_sobel_filter_manual(7))
        self.binary_button.grid(row=0, column=2, padx=5)

    def open_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")])
        if self.image_path:
            self.image = Image.open(self.image_path)
            self.display_image()

    def display_image(self):
        if self.image_path:
            self.image = self.image.resize((600, 600), Image.Resampling.LANCZOS)
            self.tk_image = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(300, 300, image=self.tk_image)

    def apply_sobel_filter_manual(self, size=3):
        if self.image is None:
            return

        # Преобразуем PIL-изображение в массив numpy в оттенках серого
        image_array = np.array(self.image.convert('L'))

        # Определяем ядра фильтра Собеля для горизонтального и вертикального градиента
        sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # Получаем размеры изображения
        height, width = image_array.shape

        # Создаем пустой массив для хранения результата
        sobel_image = np.zeros((height, width))

        # Применяем фильтр Собеля, игнорируя крайние пиксели
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                # Вычисляем горизонтальный и вертикальный градиенты
                gx = np.sum(np.multiply(sobel_kernel_x, image_array[y - 1:y + 2, x - 1:x + 2]))
                gy = np.sum(np.multiply(sobel_kernel_y, image_array[y - 1:y + 2, x - 1:x + 2]))

                # Вычисляем магнитуду градиента
                sobel_image[y, x] = np.sqrt(gx ** 2 + gy ** 2)

        # Нормализуем изображение
        sobel_image = np.clip(sobel_image / np.max(sobel_image) * 255, 0, 255).astype(np.uint8)

        # Преобразуем обратно в PIL-изображение
        sobel_image_pil = Image.fromarray(sobel_image)

        # Обновляем изображение для отображения
        self.tk_image = ImageTk.PhotoImage(image=sobel_image_pil)
        self.canvas.create_image(300, 300, image=self.tk_image)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
