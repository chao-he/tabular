import os
import cv2
import json
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


def blend(image, shape):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, shape)
    image = cv2.erode(image, kernel, iterations=3)
    return cv2.dilate(image, kernel, iterations=5)


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
    #image = cv2.fastNlMeansDenoising(image, h=hsize)
    #_, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
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


def get_bbox(img):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contours]


def segment(img, iteration=10):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))
    img = cv2.dilate(img, kernel, iterations=3)

    boxes, canvas = get_bbox(img), np.zeros_like(img)
    for i in range(iteration):
        for box in boxes:
            cv2.rectangle(canvas, box, 255, -1)
        n, boxes = len(boxes), get_bbox(canvas)
        if len(boxes) == n:
            break
    return boxes


def normalize(text):
    return text.split()[0].lower().strip(".").strip("#")


def check_title(table):
    if not table:
        return 0
    font_width, (x0, x1, title) = 10, table[0][2][0]
    for ix in range(1, len(table)):
        xmin, xmax, txt = table[ix][2][0]
        if xmin-font_width < x0 or xmax-xmin+font_width >= x1-x0:
            continue
        return ix
    return 0


def check_header(table):
    first_column = [(i, normalize(d[2][0][2])) for i,d in enumerate(table) if len(d[2]) > 3]
    for ix, word in first_column:
        if word in ("compound", "entry", "structure", "fragment"):
            return ix
    for ix, word in first_column:
        if word[:4] in ("comp", "cmpd", "cpd"):
            return ix
    for ix, word in first_column:
        if word in ("id", "no",  "ex"):
            return ix
    return -1

def process(imgfile, margin, debug=False):
    origin_img = cv2.imread(imgfile)

    gray = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    gray = remove_lines(255 - gray)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    gray = cv2.dilate(gray, kernel, iterations=3)
    gray = cv2.erode(gray, kernel, iterations=3)

    layout = json.load(open(imgfile.replace(".full.png", ".json")))

    scaff_ix = max(0, check_title(layout))
    table_ix = max(scaff_ix, check_header(layout))

    for y0, y1, row in layout[:scaff_ix]:
        gray[y0:y1,:] = 0

    table_start = layout[table_ix][0]
    for y0, y1, row in layout[table_ix:]:
        for x0, x1, txt in row:
            gray[max(0,y0-1):y1+1,max(0,x0-1):x1+1] = 0

    txt_boxes, s_boxes, r_boxes = [], [], []

    for y0, y1, row in layout:
        for x0, x1, txt in row:
            txt_boxes.append([x0, y0, x1-x0, y1-y0])

    boxes_a = segment(gray[:table_start,:])
    boxes_b = segment(gray[table_start:,:])

    for x,y,w,h in boxes_a:
        if w > 50 and h > 50:
            s_boxes.append([x,y,w,h])

    for x,y,w,h in boxes_b:
        if w > 50 and h > 50:
            r_boxes.append([x,y+table_start,w,h])

    for i in range(len(r_boxes)):
        for j in range(len(txt_boxes)):
            xa,ya,wa,ha = r_boxes[i]
            xb,yb,wb,hb = txt_boxes[j]
            if check_intersection(r_boxes[i], txt_boxes[j], margin):
                x1,x2 = min(xa, xb), max(xa+wa, xb+wb)
                y1,y2 = min(ya, yb), max(ya+ha, yb+hb)
                r_boxes[i] = [x1, y1, x2-x1, y2-y1]
                txt_boxes[j] = [0,0,0,0]


    for box in s_boxes:
        cv2.rectangle(origin_img, box, (0,0,255), 2)

    for box in r_boxes:
        cv2.rectangle(origin_img, box, (0,255,0), 2)

    for box in txt_boxes:
        cv2.rectangle(origin_img, box, (128,128,128), 1)

    cv2.imwrite(imgfile.replace(".full.png", ".seg.png"), origin_img)
    print(imgfile)


if __name__ == "__main__":
    from config import DOC_ROOT
    for tblroot in glob(f"{DOC_ROOT}/*"):
        for imgfile in sorted(glob(f"{tblroot}/*.full.png")):
            process(imgfile, 10, False)
