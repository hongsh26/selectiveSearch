import selectivesearch
import cv2
import matplotlib.pyplot as plt
import numpy as np


def showImg(img):
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.show()


def compute_iou(cand_box, gt_box):
    # Calculate intersection areas
    x1 = np.maximum(cand_box[0], gt_box[0])
    y1 = np.maximum(cand_box[1], gt_box[1])
    x2 = np.minimum(cand_box[2], gt_box[2])
    y2 = np.minimum(cand_box[3], gt_box[3])

    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)

    cand_box_area = (cand_box[2] - cand_box[0]) * (cand_box[3] - cand_box[1])
    gt_box_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    union = cand_box_area + gt_box_area - intersection

    iou = intersection / union
    return iou

def find_box(path,box_color,answer_box):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    _, regions = selectivesearch.selective_search(img, scale=100, min_size=200)
    rect_where = [regrion['rect'] for regrion in regions if regrion['size'] > 6000]
    for index, cand_box in enumerate(rect_where):
        cand_box = list(cand_box)
        cand_box[2] += cand_box[0]
        cand_box[3] += cand_box[1]

        for where in answer_box:
            iou = compute_iou(cand_box, where)
            cv2.rectangle(img, (where[0], where[1]), (where[2], where[3]), color=(10, 10, 400),
                          thickness=2)
            if iou > 0.6:
                print('index:', index, "iou:", iou, 'rectangle:', (cand_box[0], cand_box[1], cand_box[2], cand_box[3]))
                cv2.rectangle(img, (cand_box[0], cand_box[1]), (cand_box[2], cand_box[3]), color=box_color,
                              thickness=1)
                text = "{}: {:.2f}".format(index, iou)
                cv2.putText(img, text, (cand_box[0] + 100, cand_box[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            color=box_color, thickness=1)
    showImg(img)



one_path = 'one.png'
three_path = 'three.jpeg'
box_color = (125, 255, 51)
answer_box = [65, 15, 450, 600]

answer_box1 = [500, 350, 1100, 1250]
answer_box2 = [990, 410, 1300, 650]
answer_box3 = [770, 600, 1400, 1250]

box1 = [answer_box]
box2 = [answer_box1, answer_box2, answer_box3]
find_box(one_path, box_color, box1)
find_box(three_path, box_color, box2)