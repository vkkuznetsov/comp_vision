from ultralytics import YOLO


from PIL import Image
import glob
import os


# Функция для загрузки изображений и аннотаций
def load_data(image_folder, label_folder):
    images = []
    labels = []
    image_files = glob.glob(os.path.join(image_folder, "*.jpg"))
    for image_file in image_files:
        label_file = os.path.join(label_folder, os.path.basename(image_file).replace(".jpg", ".txt"))
        if os.path.exists(label_file):
            images.append(image_file)
            labels.append(label_file)
    return images, labels


# Функция для загрузки изображения и его аннотации
def load_image_and_label(image_file, label_file):
    image = Image.open(image_file).convert("RGB")
    with open(label_file, "r") as f:
        label = [list(map(float, line.strip().split())) for line in f]
    return image, label


# Load a model

model = YOLO("yolov8m.pt")



# Преобразование изображений для модели
transform = transforms.Compose([
    transforms.ToTensor(),
])


# Функция для вычисления IoU
def bbox_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area
    return iou


# Функция для вычисления точности
def evaluate_model(images, labels):
    total = 0
    correct = 0
    iou_threshold = 0.5  # Порог IoU для определения правильности предсказания

    for image_file, label_file in zip(images, labels):
        image, label = load_image_and_label(image_file, label_file)
        image_tensor = transform(image).unsqueeze(0)
        preds = model(image_tensor)

        # Преобразование предсказаний в нормализованный формат
        pred_boxes = []
        for p in preds.xywh[0]:
            class_id = int(p[5])
            x_center = p[0].item() / image.width
            y_center = p[1].item() / image.height
            width = p[2].item() / image.width
            height = p[3].item() / image.height
            pred_boxes.append([class_id, x_center, y_center, width, height])

        # Сравнение предсказаний с аннотациями
        for true_box in label:
            true_class_id, true_x, true_y, true_w, true_h = true_box
            true_box_coords = [true_x, true_y, true_w, true_h]
            best_iou = 0
            for pred_box in pred_boxes:
                if pred_box[0] == true_class_id:
                    pred_box_coords = pred_box[1:]
                    iou = bbox_iou(true_box_coords, pred_box_coords)
                    if iou > best_iou:
                        best_iou = iou
            if best_iou > iou_threshold:
                correct += 1
        total += len(label)

    accuracy = correct / total
    return accuracy


# Пример использования
image_folder = "path/to/images"
label_folder = "path/to/labels"
images, labels = load_data(image_folder, label_folder)
accuracy = evaluate_model(images, labels)
print(f"Model Accuracy: {accuracy:.2f}")
