import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2


class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Сегментация изображения")

        self.image_path = None
        self.image = None
        self.tk_image = None

        self.create_menu()
        self.create_image_loading_interface()
        self.create_image_display_interface()
        self.create_threshold_interface()

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

        self.threshold_scale = tk.Scale(self.threshold_frame, from_=0, to=100, orient=tk.HORIZONTAL, label="Процентиль")
        self.threshold_scale.set(95)
        self.threshold_scale.pack()

        self.segment_button = tk.Button(self.root, text="Применить сегментацию", command=self.apply_segmentation)
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

    def apply_segmentation(self):
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


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
