import numpy as np
import cv2
import matplotlib.pyplot as plt

def mser_image_processing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    visual = image.copy()
 
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    #cv2.polylines(visual, hulls, 1, (0, 255, 0))

    keep = []
    for c in hulls:
        x, y, w, h = cv2.boundingRect(c)
        keep.append([x, y, x + w, y + h])
    keep = np.array(keep)
    boxes = nms(keep, 0.5)
    for box in boxes:
        cv2.rectangle(visual, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
    
    # 使用 Matplotlib 显示图像
    plt.imshow(cv2.cvtColor(visual, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # 关闭坐标轴显示
    plt.title("Text Detection")
    plt.show()

def nms(boxes, overlapThresh):
    if len(boxes) == 0:
        return []
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    return boxes[pick].astype("int")

if __name__ == '__main__':
    img = cv2.imread("./Test/img/image_925.jpg", cv2.IMREAD_COLOR)
    
    if img is not None:
        mser_image_processing(img)
    else:
        print("Image loading failed.")

#https://blog.csdn.net/wzw12315/article/details/105268971 and GPT