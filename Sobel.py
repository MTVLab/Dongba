import numpy as np
import cv2
import matplotlib.pyplot as plt
import time  # 导入time模块用于FPS计算

def parse_labels(file_path):
    with open(file_path, 'r') as file:
        label_data = file.read()
    boxes = []
    entries = label_data.strip().split("###")
    for entry in entries:
        if not entry.strip():
            continue
        points = list(map(int, entry.split(',')))
        xs = points[0::2]
        ys = points[1::2]
        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)
        boxes.append([x_min, y_min, x_max, y_max])
    return boxes

def iou(box1, box2):
    """计算两个框的交并比"""
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

def evaluate_detection(region, ground_truths, iou_threshold=0.5):
    TP = 0
    FP = 0
    detected = []

    for box in region:
        box_detected = False
        for gt in ground_truths:
            if iou(box, gt) >= iou_threshold:
                if gt not in detected:
                    TP += 1
                    detected.append(gt)
                    box_detected = True
                    break
        if not box_detected:
            FP += 1

    FN = len(ground_truths) - len(detected)
    return TP, FP, FN

def calculate_metrics(TP, FP, FN):
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f_measure = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f_measure

def traditional_image_processing(image_path, ground_truths):
    start_time = time.time()  # 记录开始时间

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print("Error: Image not found.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel = cv2.Sobel(gray, cv2.CV_8U, 0, 1, ksize=3)
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))

    dilation = cv2.dilate(binary, element2, iterations=1)
    erosion = cv2.erode(dilation, element1, iterations=1)
    dilation2 = cv2.dilate(erosion, element2, iterations=2)

    contours, hierarchy = cv2.findContours(dilation2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    region = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        x_min = min(box[:, 0])
        y_min = min(box[:, 1])
        x_max = max(box[:, 0])
        y_max = max(box[:, 1])
        region.append([x_min, y_min, x_max, y_max])

    TP, FP, FN = evaluate_detection(region, ground_truths)
    precision, recall, f_measure = calculate_metrics(TP, FP, FN)
    print(f"Precision: {precision}, Recall: {recall}, F-measure: {f_measure}")
    
    end_time = time.time()  # 记录结束时间
    fps = 1 / (end_time - start_time)  # 计算FPS
    print(f"FPS: {fps}")

    # 绘制检测到的区域
    for box in region:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)  

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

if __name__ == '__main__':
    ground_truths = parse_labels(label_data)
    traditional_image_processing("./Test/img/image_70.jpg", ground_truths)
