import os
import cv2
import numpy as np
from functools import cmp_to_key
from glob import glob


def cmp_couter(a, b):
    xa,ya,wa,ha = a
    xb,yb,wb,hb = b
    if ya+ha < yb:
        return -1
    elif ya > yb+hb:
        return 1
    else:
        return -1 if xa < xb else 1

def read_binary_image(imgfile):
    img = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    return 255-img if img is not None else None

def detect_line(image, shape):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, shape)
    image2 = cv2.erode(image, kernel, iterations=3)
    return cv2.dilate(image2, kernel, iterations=5)

def remove_lines(image, lpad=50, ksize=4, hsize=50):
    xl = detect_line(image, (min(lpad, image.shape[1]//10), 1))
    yl = detect_line(image, (1, min(lpad, image.shape[0]//10)))

    lines = cv2.addWeighted(xl, 0.5, yl, 0.5, 0)
    _, lines = cv2.threshold(lines, 128, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    lines = cv2.erode(~lines, kernel, iterations=2)
    _, lines = cv2.threshold(lines, 128, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

    image = cv2.bitwise_and(image, lines)
    image = cv2.fastNlMeansDenoising(image, h=hsize)
    _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    return image


def check_intersection(boxA, boxB, margin):
    x = max(boxA[0], boxB[0])
    y = max(boxA[1], boxB[1])
    w = min(boxA[0] + boxA[2], boxB[0] + boxB[2]) - x
    h = min(boxA[1] + boxA[3], boxB[1] + boxB[3]) - y
    return w > -margin and h > -margin


def merge_box(boxes, margin):
    removed, added = set(), []
    for i in range(len(boxes)):
        for j in range(i+1, len(boxes)):
            if check_intersection(boxes[i], boxes[j], margin):
                xa,ya,wa,ha = boxes[i]
                xb,yb,wb,hb = boxes[j]
                x1,x2 = min(xa, xb), max(xa+wa, xb+wb)
                y1,y2 = min(ya, yb), max(ya+ha, yb+hb)
                added.append((x1, y1, x2-x1, y2-y1))
                removed.add(i)
                removed.add(j)
    for i,b in enumerate(boxes):
        if i not in removed:
            added.append(b)
    return added


def get_bbox(img, margin):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return merge_box([cv2.boundingRect(c) for c in contours], margin)


def segment(img, margin=5, iteration=10):
    boxes, canvas = get_bbox(img, margin), np.zeros_like(img)
    for i in range(iteration):
        for box in boxes:
            cv2.rectangle(canvas, box, 255, -1)
        n, boxes = len(boxes), get_bbox(canvas, margin)
        if len(boxes) == n:
            break
    return boxes


def process(imgfile, margin, debug=False):
    origin_img = cv2.imread(imgfile)

    gray = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    gray = remove_lines(255 - gray)

    boxes = segment(gray, margin)
    target, area = -1, 0
    for i, (x,y,w,h) in enumerate(boxes):
        if w>100 and h>100:
            if area < w*h:
                area = w*h
                target = i

    if target >= 0:
        x,y,w,h = boxes[target]
        cv2.imwrite(imgfile.replace(".full.png", ".mol.png"), origin_img[y:y+h,x:x+w])

    if debug or target == -1:
        for box in boxes:
            cv2.rectangle(origin_img, box, (0,255,0), 1)
        cv2.imwrite(imgfile.replace(".full.png", ".seg.png"), origin_img)
    print(imgfile, target)


if __name__ == "__main__":
    docroot = os.path.expanduser("~/dev/tabular/tables")
    for tblroot in glob(f"{docroot}/*"):
        for imgfile in sorted(glob(f"{tblroot}/*.001.full.png")):
            process(imgfile, 10, False)
