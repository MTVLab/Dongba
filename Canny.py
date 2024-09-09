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

def canny_image_processing(image):
    if image is None:
        print("Error: Image not found.")
        return []
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 应用高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # 运行 Canny 算子
    canny = cv2.Canny(blurred, 50, 150)
    # 寻找轮廓
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10 and h > 10:  # 过滤掉太小的区域
            boxes.append([x, y, x + w, y + h])
    return boxes

def visualize_image(image_path, ground_truth_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print("Image not found:", image_path)
        return
    ground_truths = parse_labels(ground_truth_path)
    detected_boxes = canny_image_processing(image)

    # # 绘制真实框
    # for box in ground_truths:
    #     cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)  # 红色表示真实框

    # 绘制检测框
    for box in detected_boxes:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)  # 绿色表示检测框

    # 使用 Matplotlib 显示图像
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Text Detection with Canny Edge Detection')
    plt.axis('off')  # 关闭坐标轴显示
    plt.show()

if __name__ == '__main__':
    img_directory = "./Test/img/"
    gt_directory = "./Test/gt/"
    image_path = os.path.join(img_directory, "image_925.jpg")
    ground_truth_path = os.path.join(gt_directory, "gt_image_338.txt")
    visualize_image(image_path, ground_truth_path)
