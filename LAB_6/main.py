import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk


class MotionDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Motion Detection App")

        # Кнопка для загрузки видео
        self.load_button = tk.Button(self.root, text="Загрузить видео", command=self.load_video)
        self.load_button.grid(row=0, column=0, pady=10, padx=10)

        # Поля для ввода параметров
        self.lambda_label = tk.Label(self.root, text="λ (весовой коэффициент):")
        self.lambda_label.grid(row=1, column=0, padx=10)
        self.lambda_entry = tk.Entry(self.root)
        self.lambda_entry.grid(row=1, column=1, padx=10)

        self.threshold_label = tk.Label(self.root, text="Порог (T):")
        self.threshold_label.grid(row=2, column=0, padx=10)
        self.threshold_entry = tk.Entry(self.root)
        self.threshold_entry.grid(row=2, column=1, padx=10)

        self.neighborhood_label = tk.Label(self.root, text="Размер окрестности (k):")
        self.neighborhood_label.grid(row=3, column=0, padx=10)
        self.neighborhood_entry = tk.Entry(self.root)
        self.neighborhood_entry.grid(row=3, column=1, padx=10)

        # Кнопки для запуска алгоритмов
        self.horn_schunck_button = tk.Button(self.root, text="Алгоритм Хорна-Шанка", command=self.run_horn_schunck)
        self.horn_schunck_button.grid(row=4, column=0, pady=10, padx=10)

        self.lucas_kanade_button = tk.Button(self.root, text="Алгоритм Лукаса-Канаде", command=self.run_lucas_kanade)
        self.lucas_kanade_button.grid(row=4, column=1, pady=10, padx=10)

        # Предпросмотр видео
        self.video_preview_label = tk.Label(self.root)
        self.video_preview_label.grid(row=5, column=0, columnspan=2, pady=10)

        # Кнопки управления видео
        self.play_button = tk.Button(self.root, text="Воспроизвести", command=self.play_video)
        self.play_button.grid(row=6, column=0, pady=5, padx=10)

        self.pause_button = tk.Button(self.root, text="Пауза", command=self.pause_video)
        self.pause_button.grid(row=6, column=1, pady=5, padx=10)

        self.forward_button = tk.Button(self.root, text="Вперёд", command=self.forward_video)
        self.forward_button.grid(row=7, column=0, pady=5, padx=10)

        self.backward_button = tk.Button(self.root, text="Назад", command=self.backward_video)
        self.backward_button.grid(row=7, column=1, pady=5, padx=10)

        # Переменные состояния видео
        self.playing = False
        self.paused = True
        self.prev_frame = None

    def load_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
        self.cap = cv2.VideoCapture(self.video_path)
        self.show_frame()

    def show_frame(self):
        if not self.paused:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (640, 480))  # Изменение размера для отображения на экране
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                self.video_preview_label.config(image=self.photo)
                self.video_preview_label.image = self.photo
                self.prev_frame = frame
                self.root.after(10, self.show_frame)
            else:
                self.cap.release()

    def play_video(self):
        if not self.playing:
            self.playing = True
            self.paused = False
            self.show_frame()

    def pause_video(self):
        if self.playing:
            self.paused = not self.paused
            self.playing = not self.playing

    def forward_video(self):
        if self.playing:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.cap.get(cv2.CAP_PROP_POS_FRAMES) + 100)

    def backward_video(self):
        if self.playing:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.cap.get(cv2.CAP_PROP_POS_FRAMES) - 100)

    def run_horn_schunck(self):
        lambda_value = float(self.lambda_entry.get())
        threshold_value = float(self.threshold_entry.get())

        if hasattr(self, 'video_path') and self.prev_frame is not None:
            # Преобразуйте предыдущий и текущий кадры в градации серого
            prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
            ret, curr_frame = self.cap.read()
            if ret:
                curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

                # Применяем алгоритм Хорна-Шанка
                flow = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, lambda_value,
                                                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

                # Преобразуем поток в изображение для визуализации
                hsv = np.zeros_like(self.prev_frame)
                hsv[..., 1] = 255
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                result_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                # Показываем обработанный кадр
                self.show_processed_frame(result_frame)

    def run_lucas_kanade(self):
        neighborhood_value = int(self.neighborhood_entry.get())

        if hasattr(self, 'video_path') and self.prev_frame is not None:
            # Преобразуйте предыдущий и текущий кадры в градации серого
            prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
            ret, curr_frame = self.cap.read()
            if ret:
                curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.resize(curr_gray, (640, 480))
                # Определите характеристики для отслеживания
                height, width = prev_gray.shape
                p0 = np.array([[x, y] for x in range(width) for y in range(height)], dtype=np.float32)

                # Применяем алгоритм Лукаса-Канаде
                p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None,
                                                       winSize=(neighborhood_value, neighborhood_value), maxLevel=4)

                # Отфильтруем допустимые точки и рисуем поток
                for new, old in zip(p1.reshape(-1, 2), p0.reshape(-1, 2)):
                    a, b = new
                    c, d = old
                    if abs(a - c) < 7 and abs(b - d) < 7:
                        curr_gray = cv2.line(curr_gray, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 1)
                        curr_gray = cv2.circle(curr_gray, (int(a), int(b)), 2, (0, 255, 0), -1)

                # Показываем обработанный кадр
                self.show_processed_frame(curr_gray)

    def show_processed_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (640, 480))  # Изменение размера для отображения на экране
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        self.video_preview_label.config(image=self.photo)
        self.video_preview_label.image = self.photo


def main():
    root = tk.Tk()
    app = MotionDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
