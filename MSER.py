import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import os

def parse_labels(file_path):
    with open(file_path, 'r') as file:
        label_data = file.read()
    boxes = []
    entries = label_data.strip().split("###")
    for entry in entries:
        if not entry.strip():
            continue
        points = [int(num) for num in ''.join(entry.split()).split(',') if num]
        xs = points[0::2]
        ys = points[1::2]
        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)
        boxes.append([x_min, y_min, x_max, y_max])
    return boxes

def iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area

def mser_image_processing(image):
    if image is None:
        print("Error: Image not found.")
        return []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    boxes = []
    for c in hulls:
        x, y, w, h = cv2.boundingRect(c)
        boxes.append([x, y, x + w, y + h])
    return boxes

def visualize_image(image_path, ground_truth_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print("Image not found:", image_path)
        return
    ground_truths = parse_labels(ground_truth_path)
    detected_boxes = mser_image_processing(image)

    # # 绘制真实框
    # for box in ground_truths:
    #     cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)  # 蓝色表示真实框

    # 绘制检测框
    for box in detected_boxes:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)  # 绿色表示检测框

    # 使用 Matplotlib 显示图像
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Text Detection')
    plt.axis('off')  # 关闭坐标轴显示
    plt.show()

if __name__ == '__main__':
    img_directory = "./Test/img/"
    gt_directory = "./Test/gt/"
    image_path = os.path.join(img_directory, "image_925.jpg")
    ground_truth_path = os.path.join(gt_directory, "gt_image_70.txt")
    visualize_image(image_path, ground_truth_path)
