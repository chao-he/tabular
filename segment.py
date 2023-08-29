import os
import re
import cv2
import json
import numpy as np
from glob import glob


def read_binary_image(imgfile):
    img = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    return 255-img if img is not None else None


def blur(image, ksize):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize,ksize))
    image = cv2.dilate(image, kernel, iterations=3)
    return cv2.erode(image, kernel, iterations=3)


def denoise(image, kshape):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kshape)
    image2 = cv2.erode(image, kernel, iterations=3)
    return cv2.dilate(image2, kernel, iterations=5)


def remove_lines(image, lpad=50, ksize=4, hsize=50):
    xl = denoise(image, (min(lpad, image.shape[1]//10), 1))
    yl = denoise(image, (1, min(lpad, image.shape[0]//10)))

    lines = cv2.addWeighted(xl, 0.5, yl, 0.5, 0)
    _, lines = cv2.threshold(lines, 128, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    lines = cv2.erode(~lines, kernel, iterations=2)
    _, lines = cv2.threshold(lines, 128, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

    return cv2.bitwise_and(image, lines)
    #image = cv2.fastNlMeansDenoising(image, h=hsize)
    #_, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    #return image


def read_image(imgfile):
    origin_img = cv2.imread(imgfile)
    if origin_img is not None:
        gray = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)
        _, gray = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        gray = remove_lines(255 - gray)
        return blur(gray, 2)


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

def scanline(boxes, axis=0):
    segs, start, end = [], 0, 0
    for b in sorted(boxes, key=lambda b: b[axis]):
        s, t = b[axis], b[axis] + b[axis+2]
        if s > end:
            segs.append([start, end])
            start, end = s, t
        else:
            end = max(end, t)
    segs.append([start, end])
    return [s for s in segs if s[1]-s[0]>0]

def locate(xmin, xmax, lines):
    for i in range(1, len(lines)):
        (s0,t0), (s1,t1) = lines[i-1], lines[i]
        if s0 <= xmin < s1:
            if xmax <= t0 or xmin <= t0 <= xmax:
                return i-1
            elif xmax >= s1:
                return i
            else: #t0 < xmin, xmax < s1
                return i-1
    return len(lines)-1

def normalize(text):
    return text.split()[0].lower().strip(".").strip("#")


def normalize_text(txt):
    if re.match('^[0-9.]+\s+[0-9.]+[abc]?$', txt):
        a, b = txt.split()
        txt = f"{a}±{b}"
    txt = re.sub(u"[1I]C[5sS][0oO]", "IC50", txt)
    syns = {"R'": "R1"}
    return syns.get(txt, txt)

def find_first_of(tags, pat):
    for ix, tag in tags:
        if tag in pat:
            return ix


def nonzero(flags):
    i, e = 0, len(flags)
    while i < len(flags):
        while i < e and flags[i] == 0:
            i += 1
        s = i
        while i < e and flags[i] == 1:
            i += 1
        if s < i:
            yield s,i

def aligment(boxes):
    width = np.max([b[2] for b in boxes]) if boxes else 0
    height = np.max([b[3] for b in boxes]) if boxes else 0

    flags = np.zeros(height)
    for x1,y1,x2,y2,t in boxes:
        flags[y1:y2] = 1

    rows = []
    for ymin, ymax in nonzero(flags):
        cells, mask = [], np.zeros(width)
        for x,y,w,h,t in boxes:
            if y>=ymin and h<=ymax:
                mask[x:w] = 1
                cells.append([x,y,w,h,t])
        columns = []
        for xmin, xmax in nonzero(mask):
            column = [(y, t) for x,y,w,h,t in cells if x>=xmin and w<=xmax]
            text = " ".join([t for y, t in sorted(column)])
            columns.append([xmin, xmax, text])
        rows.append([ymin, ymax, columns])
    return rows


def check_table_position(table):
    group_of_tags = [
        ("compound", "compounds"),
        ("entry", "structure", "fragment"),
        ("comp", "compd", "cmpd", "cpd"),
        ("compa", "compb", "compc", "compda", "compdb", "compdc", "compdd"),
        ("id", "no",  "ex")
    ]

    left, right = 0, 0
    for _, _, row in table:
        left, right = min(left, row[0][0]), max(right, row[-1][1])
    table_width = right - left

    title_last_row, table_first_row, table_last_row = -1, -1, -1

    title_h = table[0][1] - table[0][0]
    for i, (ymin,ymax,row) in enumerate(table):
        xmin, xmax, txt = row[0]
        w, h = xmax-xmin, ymax-ymin
        if h < 1.1*title_h and xmin < left+10 and (len(row)<=2 or w>table_width*0.65):
            title_last_row = i
        break

    for i, (ymin,ymax,row) in enumerate(table):
        if i > title_last_row and len(row) > 2:
            table_first_row = i

    first_column = []
    for i, (ymin,ymax,row) in enumerate(table):
        if i > title_last_row and len(row) > 1:
            first_column.append([i, normalize(row[0][2])])

    for tags in group_of_tags:
        row = find_first_of(first_column, tags)
        if row is not None:
            table_first_row = row
            break

    table_last_row = table_first_row
    for i, (ymin,ymax,row) in enumerate(table):
        if i > table_first_row:
            xmin, xmax, txt = row[0]
            if xmin < left+10 and len(row) < 3:
                break
            table_last_row = i

    return title_last_row, table_first_row, table_last_row + 1


def read_layout(layout_path):
    layout = json.load(layout_path)
    layout = aligment(layout)

    title_ix, table_ix, note_ix = check_table_position(layout)

    t_boxes = []
    for y0, y1, row in layout[table_ix:note_ix]:
        for x0, x1, txt in row:
            t_boxes.append([x0, y0, x1-x0, y1-y0, txt])

    return layout, t_boxes, title_ix, table_ix, note_ix


def process(imgfile, margin, debug=False):
    layout_path = imgfile.replace(".full.png", ".json")
    layout, t_boxes, title_ix, table_ix, note_ix = read_layout(open(layout_path))

    if title_ix < 0 or table_ix < 0:
        return

    image = read_image(imgfile)
    for i, (ymin, ymax, row) in enumerate(layout):
        if i <= title_ix or i >= table_ix:
            for xmin, xmax, txt in row:
                x0 = max(0, xmin-1)
                y0 = max(0, ymin-1)
                x1 = min(image.shape[1], xmax+1)
                y1 = min(image.shape[0], ymax+1)
                image[y0:y1,x0:x1] = 0

    title = []
    for i in range(title_ix+1):
        ymin, ymax, row = layout[i]
        for xmin, xmax, txt in row:
            title.append(txt)
    title = " ".join(title)

    note = []
    for i in range(note_ix, len(layout)):
        ymin, ymax, row = layout[i]
        for xmin, xmax, txt in row:
            note.append(txt)
    note = " ".join(note)

    title_y1 = layout[title_ix][1]
    table_y0 = layout[table_ix][0]
    table_y1 = layout[note_ix][0] if note_ix < len(layout) else image.shape[0]

    s_boxes_temp = segment(image[title_y1:table_y0,])
    r_boxes_temp = segment(image[table_y0:table_y1,])

    s_boxes = []
    for i,(x,y,w,h) in enumerate(s_boxes_temp):
        y += title_y1
        if w > 50 and h > 50:
            s_boxes.append((x,y,w,h))

    image = read_binary_image(imgfile)
    r_boxes = []
    for x,y,w,h in r_boxes_temp:
        y += table_y0
        if w > 50 and h > 50:
            i, j = 0, 0
            for i in range(w):
                if np.sum(image[y:y+h,x+i]) > 0:
                    break
            for j in range(w):
                if np.sum(image[y:y+h,x+w-j]) > 0:
                    break
            r_boxes.append((x+i,y,w-i-j,h))

    table = maketable(r_boxes, t_boxes)

    img = cv2.imread(imgfile)
    for x,y,w,h in s_boxes:
        cv2.rectangle(img, (x,y,w,h), (0,0,255), 2)
    for x,y,w,h in r_boxes:
        cv2.rectangle(img, (x,y,w,h), (0,255,0), 2)
    for x,y,w,h,t in t_boxes:
        cv2.rectangle(img, (x,y,w,h), (255,0,0), 1)
    cv2.imwrite(imgfile.replace(".full.png", ".dbg.png"), img)

    return table, title, note, r_boxes


def maketable(r_boxes, t_boxes):
    removed = set()
    for i in range(len(r_boxes)):
        for j in range(len(t_boxes)):
            xa,ya,wa,ha = r_boxes[i][:4]
            xb,yb,wb,hb = t_boxes[j][:4]
            if check_intersection(r_boxes[i], t_boxes[j], 1):
                x1,x2 = min(xa, xb), max(xa+wa, xb+wb)
                y1,y2 = min(ya, yb), max(ya+ha, yb+hb)
                r_boxes[i] = [x1, y1, x2-x1, y2-y1]
                removed.add(j)

    shl = scanline(r_boxes, axis=1)
    svl = scanline(r_boxes, axis=0)
    for k, (x,y,w,h) in enumerate(r_boxes):
        j = locate(x,x+w,svl)
        x0, x1 = svl[j]
        i = locate(y,y+h,shl)
        y0, y1 = shl[i]
        r_boxes[k] = [x0,y0,x1-x0,y1-y0]

    for j in reversed(sorted(removed)):
        del t_boxes[j]

    all_boxes = t_boxes + r_boxes
    hl = scanline(t_boxes + r_boxes, axis=1)
    vl = scanline(t_boxes + r_boxes, axis=0)

    table = [[[] for j in range(len(vl))] for i in range(len(hl))]

    for x,y,w,h,t in t_boxes:
        i = locate(y,y+h,hl)
        j = locate(x,x+w,vl)
        table[i][j].append(normalize_text(t))

    for x,y,w,h in r_boxes:
        i = locate(y,y+h,hl)
        j = locate(x,x+w,vl)
        table[i][j].append("[CCO]")

    for i in range(len(table)):
        for j in range(len(table[i])):
            table[i][j] = ";".join(table[i][j])

    return table
