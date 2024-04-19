import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt


class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing App")

        self.image_path = None

        self.create_menu()

        self.create_image_loading_interface()

        self.create_image_display_interface()

        self.create_colorfulness_interface()

        self.create_smoothing_interface()

        self.create_sharpness_interface()

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

        self.sharpness_label = tk.Label(self.root, text="Резкость: Н/Д")
        self.sharpness_label.pack(pady=10)

    def create_image_display_interface(self):
        self.canvas = tk.Canvas(self.root, width=200, height=200)
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

        self.clip_button = tk.Button(self.colorfulness_frame, text="Вырезание диапозона(исх)",
                                     command=self.clip_range_original)
        self.clip_button.grid(row=0, column=4, padx=5)

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

    def create_sharpness_interface(self):
        self.sharpness_frame = tk.LabelFrame(self.root, text="Резкость")
        self.sharpness_frame.pack(pady=10)

        self.unsharp_mask_button = tk.Button(self.sharpness_frame, text="Нерезкое маскирование",
                                             command=self.apply_unsharp_mask)
        self.unsharp_mask_button.grid(row=0, column=0, padx=5)

        self.kernel_size_entry = tk.Entry(self.sharpness_frame)
        self.kernel_size_entry.grid(row=1, column=1, padx=5)
        self.kernel_size_label = tk.Label(self.sharpness_frame, text="Размер ядра:")
        self.kernel_size_label.grid(row=1, column=0, padx=5)

        self.amount_entry = tk.Entry(self.sharpness_frame)
        self.amount_entry.grid(row=2, column=1, padx=5)
        self.amount_label = tk.Label(self.sharpness_frame, text="Коэффициент резкости:")
        self.amount_label.grid(row=2, column=0, padx=5)

        self.compare_filters_button = tk.Button(self.sharpness_frame, text="Сравнить фильтры",
                                                command=self.compare_sharpness_transformations)
        self.compare_filters_button.grid(row=3, column=0, padx=5)

    def sigma_filter(self, sigma=1):
        image = Image.open(self.image_path)
        image_array = np.array(image)

        filtered_image_array = np.zeros_like(image_array, dtype=np.float32)
        height, width, channels = image_array.shape

        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    y_min = max(0, y - sigma)
                    y_max = min(height - 1, y + sigma)
                    x_min = max(0, x - sigma)
                    x_max = min(width - 1, x + sigma)

                    window = image_array[y_min:y_max + 1, x_min:x_max + 1, c]
                    average_value = np.mean(window)

                    filtered_image_array[y, x, c] = average_value

        filtered_image = Image.fromarray(np.uint8(filtered_image_array))
        return filtered_image

    def gaussian_blur(self, sigma, image_array=None):
        if image_array is None:
            image = Image.open(self.image_path)
            image_array = np.array(image)

        kernel_size = 6 * sigma + 1
        kernel = self.gaussian_kernel(kernel_size, sigma)

        blurred_image_array = self.apply_convolution(image_array, kernel)

        blurred_image = Image.fromarray(np.uint8(blurred_image_array))

        return blurred_image

    def apply_convolution(self, image_array, kernel):
        output = np.zeros_like(image_array)
        for c in range(image_array.shape[2]):
            output[:, :, c] = convolve2d(image_array[:, :, c], kernel, mode='same', boundary='symm')
        return output

    def gaussian_kernel(self, size, sigma=1.0):
        x, y = np.meshgrid(np.arange(-size // 2 + 1, size // 2 + 1),
                           np.arange(-size // 2 + 1, size // 2 + 1))
        kernel = np.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))
        return kernel / np.sum(kernel)

    def display_blur_images(self, mode):
        sigmas = [1, 3, 10]

        fig, axs = plt.subplots(3, len(sigmas), figsize=(15, 10))

        for i, sigma in enumerate(sigmas):
            if mode == 'gaus':
                original_image = Image.open(self.image_path)
                blurred_image = self.gaussian_blur(sigma)
            elif mode == 'sigma':
                original_image = Image.open(self.image_path)
                blurred_image = self.sigma_filter(sigma)

            axs[0, i].imshow(original_image)
            axs[0, i].set_title(f"Original (Sigma = {sigma})")
            axs[0, i].axis('off')

            axs[1, i].imshow(blurred_image)
            axs[1, i].set_title(f"Blurred (Sigma = {sigma})")
            axs[1, i].axis('off')

            diff_array = np.abs(np.array(original_image) - np.array(blurred_image))

            axs[2, i].imshow(diff_array)
            axs[2, i].set_title(f"Difference (Sigma = {sigma})")
            axs[2, i].axis('off')

        plt.tight_layout()
        plt.show()

    def open_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")])
        if self.image_path:
            image = Image.open(self.image_path)
            self.display_image()

            image_array = np.array(image.convert('L'))
            sharpness_value = self.assess_sharpness(image_array)
            self.sharpness_label.config(text=f"Резкость: {sharpness_value:.2f}")

    def display_image(self):
        image = Image.open(self.image_path)
        image.thumbnail((800, 800))
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
        image_array = np.array(image)
        pad_size = kernel_size // 2
        padded_image = np.pad(image_array, [(pad_size, pad_size), (pad_size, pad_size), (0, 0)], mode='edge')

        filtered_image = np.zeros_like(image_array)

        for i in range(image_array.shape[0]):
            for j in range(image_array.shape[1]):
                for k in range(image_array.shape[2]):
                    window = padded_image[i:i + kernel_size, j:j + kernel_size, k]
                    filtered_image[i, j, k] = np.mean(window)

        return Image.fromarray(filtered_image)

    def apply_median_filter(self, image, kernel_size):
        image_array = np.array(image)
        pad_size = kernel_size // 2
        padded_image = np.pad(image_array, [(pad_size, pad_size), (pad_size, pad_size), (0, 0)], mode='edge')

        filtered_image = np.zeros_like(image_array)

        for i in range(image_array.shape[0]):
            for j in range(image_array.shape[1]):
                for k in range(image_array.shape[2]):
                    window = padded_image[i:i + kernel_size, j:j + kernel_size, k]
                    filtered_image[i, j, k] = np.median(window)

        return Image.fromarray(filtered_image)

    # п.1
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
            gamma = 1.5
            c = 1
            image = Image.open(self.image_path)
            image_array = np.array(image)
            power_transformed = c * np.power(image_array, gamma)
            power_transformed = np.uint8(power_transformed)
            power_image = Image.fromarray(power_transformed)
            self.display_transformed_image(power_image)

    def binary_transform(self):
        if self.image_path:
            threshold = 128
            image = Image.open(self.image_path)
            image_array = np.array(image)
            binary_transformed = np.where(image_array < threshold, 0, 255)
            binary_image = Image.fromarray(np.uint8(binary_transformed))
            self.display_transformed_image(binary_image)

    def clip_range(self):
        if self.image_path:
            min_val = 50
            max_val = 200
            constant_value = 100
            image = Image.open(self.image_path)
            image_array = np.array(image)
            clipped_image = np.clip(image_array, min_val, max_val)
            clipped_image[np.where((image_array >= min_val) & (image_array <= max_val))] = 0
            clipped_image[np.where(image_array < min_val)] = constant_value
            clipped_image[np.where(image_array > max_val)] = constant_value
            clipped_image = np.uint8(clipped_image)
            clipped_image = Image.fromarray(clipped_image)
            self.display_transformed_image(clipped_image)

    def clip_range_original(self):
        if self.image_path:
            min_val = 50
            max_val = 200
            image = Image.open(self.image_path)
            image_array = np.array(image)
            clipped_image = np.clip(image_array, min_val, max_val)
            clipped_image[np.where((image_array >= min_val) & (image_array <= max_val))] = 0
            clipped_image = Image.fromarray(np.uint8(clipped_image))
            self.display_transformed_image(clipped_image)

    # п3.1
    def apply_unsharp_mask(self):
        kernel_size = int(self.kernel_size_entry.get() or 5)
        amount = float(self.amount_entry.get() or 1.5)
        if self.image_path:
            image = Image.open(self.image_path)
            image_array = np.array(image)
            blurred_image = self.gaussian_blur(kernel_size / 6)
            sharpened_image = image_array + amount * (image_array - blurred_image)
            sharpened_image = np.clip(sharpened_image, 0, 255).astype(np.uint8)
            self.display_transformed_image(Image.fromarray(sharpened_image))

    def compare_sharpness_transformations(self):
        if not self.image_path:
            print("No image loaded.")
            return

        kernel_sizes = [3, 5]
        amounts = [0.5, 1.0, 1.5]

        transformation_functions = {
            'Логарифмическая': self.logarithmic_transform_return,
            'Степенная': self.power_transform_return,
            'Бинарная': self.binary_transform_return,
            'Диапозон': self.clip_range_return,
            'Диапозон исход.': self.clip_range_original_return
        }

        fig, axs = plt.subplots(len(transformation_functions), len(kernel_sizes) * len(amounts), figsize=(15, 10))

        for i, (trans_name, trans_func) in enumerate(transformation_functions.items()):
            transformed_image = trans_func()
            transformed_array = np.array(transformed_image)

            for j, kernel_size in enumerate(kernel_sizes):
                for k, amount in enumerate(amounts):
                    sharpened_image = self.unsharp_mask(transformed_array, kernel_size, amount)
                    axs[i, j * len(amounts) + k].imshow(sharpened_image, cmap='gray')
                    axs[i, j * len(amounts) + k].set_title(f"{trans_name}\nKernel: {kernel_size}, Amount: {amount}")
                    axs[i, j * len(amounts) + k].axis('off')

        plt.tight_layout()
        plt.show()

    def unsharp_mask(self, image_array, kernel_size, amount):
        blurred_image = self.gaussian_blur(kernel_size / 6.0, image_array)
        sharpened_image = image_array + amount * (image_array - blurred_image)
        sharpened_image = np.clip(sharpened_image, 0, 255).astype(np.uint8)
        return sharpened_image

    def assess_sharpness(self, image_array):
        diff_x = np.abs(np.diff(image_array, axis=1))
        diff_y = np.abs(np.diff(image_array, axis=0))
        sharpness = (np.mean(diff_x) + np.mean(diff_y)) / 2
        return sharpness

    def display_transformed_image(self, image):
        image.thumbnail((200, 200))
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def logarithmic_transform_return(self):
        if self.image_path:
            image = Image.open(self.image_path)
            image_array = np.array(image)
            if len(image_array[0][0]) > 3:
                image_array = np.delete(image_array, 3, axis=2)
            c = 255 / np.log(1 + np.max(image_array))
            log_transformed = c * np.log(1 + image_array)
            log_transformed = np.uint8(log_transformed)
            return Image.fromarray(log_transformed)

    def power_transform_return(self):
        if self.image_path:
            gamma = 1.5
            c = 1
            image = Image.open(self.image_path)
            image_array = np.array(image)
            power_transformed = c * np.power(image_array, gamma)
            power_transformed = np.uint8(power_transformed)
            return Image.fromarray(power_transformed)

    def binary_transform_return(self):
        if self.image_path:
            threshold = 128
            image = Image.open(self.image_path)
            image_array = np.array(image)
            binary_transformed = np.where(image_array < threshold, 0, 255)
            return Image.fromarray(np.uint8(binary_transformed))

    def clip_range_return(self):
        if self.image_path:
            min_val = 50
            max_val = 200
            constant_value = 100
            image = Image.open(self.image_path)
            image_array = np.array(image)
            clipped_image = np.clip(image_array, min_val, max_val)
            clipped_image[np.where((image_array < min_val) | (image_array > max_val))] = constant_value
            return Image.fromarray(np.uint8(clipped_image))

    def clip_range_original_return(self):
        if self.image_path:
            min_val = 50
            max_val = 200
            image = Image.open(self.image_path)
            image_array = np.array(image)
            clipped_image = np.clip(image_array, min_val, max_val)
            return Image.fromarray(np.uint8(clipped_image))


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
