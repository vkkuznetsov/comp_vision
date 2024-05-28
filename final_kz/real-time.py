import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
from pytube import YouTube
import os

# Load a pretrained YOLOv8n model
model = YOLO("yolov8m.pt")

def run_inference(source):
    results = model(source)  # generator of Results objects
    for result in results:
        result.show()  # display to screen
        result.save(filename="result.jpg")  # save to disk
        show_image("result.jpg")

def show_image(image_path, max_size=(800, 600)):
    image = Image.open(image_path)
    image.thumbnail(max_size, Image.ANTIALIAS)  # Resize image while maintaining aspect ratio
    image = ImageTk.PhotoImage(image)
    panel.config(image=image)
    panel.image = image

def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        run_inference(file_path)

def open_video():
    file_path = filedialog.askopenfilename()
    if file_path:
        cap = cv2.VideoCapture(file_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            for result in results:
                img = result.plot()  # get the processed frame
                cv2.imshow('Video', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()

def download_youtube_video(url):
    yt = YouTube(url)
    stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
    video_path = stream.download()
    return video_path

def open_stream():
    stream_url = stream_entry.get()
    if "youtube.com" in stream_url:
        try:
            video_path = download_youtube_video(stream_url)
            open_video_file(video_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to download video: {e}")
    else:
        messagebox.showerror("Error", "Only YouTube URLs are supported.")

def open_video_file(file_path):
    cap = cv2.VideoCapture(file_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        for result in results:
            img = result.plot()  # get the processed frame
            cv2.imshow('Video', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

def evaluate_model():
    dir_path = filedialog.askdirectory()
    if dir_path:
        results = model.val(data=dir_path)
        show_metrics(results)

def show_metrics(results):
    metrics = f"""
    Precision: {results['metrics/precision']}
    Recall: {results['metrics/recall']}
    mAP50: {results['metrics/mAP_0.5']}
    mAP50-95: {results['metrics/mAP_0.5:0.95']}
    """
    messagebox.showinfo("Model Evaluation Metrics", metrics)

# GUI setup
root = tk.Tk()
root.title("YOLO Inference")

frame = tk.Frame(root)
frame.pack(pady=20)

panel = tk.Label(frame)
panel.pack()

btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

open_file_btn = tk.Button(btn_frame, text="Open Image", command=open_file)
open_file_btn.grid(row=0, column=0, padx=10)

open_video_btn = tk.Button(btn_frame, text="Open Video", command=open_video)
open_video_btn.grid(row=0, column=1, padx=10)

stream_entry = tk.Entry(btn_frame)
stream_entry.grid(row=0, column=2, padx=10)
stream_entry.insert(0, "Enter stream URL")

open_stream_btn = tk.Button(btn_frame, text="Open Stream", command=open_stream)
open_stream_btn.grid(row=0, column=3, padx=10)

evaluate_model_btn = tk.Button(btn_frame, text="Evaluate Model", command=evaluate_model)
evaluate_model_btn.grid(row=0, column=4, padx=10)

root.mainloop()
