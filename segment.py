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
    img = cv2.imread(imgfile)
    if img is None:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, gray = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    gray = remove_lines(255 - gray)
    return blur(gray, 2)


def check_intersection(boxA, boxB, margin):
    x = max(boxA[0], boxB[0])
    y = max(boxA[1], boxB[1])
    w = min(boxA[0] + boxA[2], boxB[0] + boxB[2]) - x
    h = min(boxA[1] + boxA[3], boxB[1] + boxB[3]) - y
    marginX, marginY = margin
    return w > -marginX and h > -marginY


def merge_box_s(boxes, margin):
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
        ("id", "no",  "ex"),
        ("assay", "species")
    ]

    left, right = 0, 0
    for _, _, row in table:
        left, right = min(left, row[0][0]), max(right, row[-1][1])
    table_width = right - left

    title_last_row, table_first_row, table_last_row = 0, -1, -1

    th = table[0][1] - table[0][0]
    tw = max(table[0][2][0][1] - table[0][2][0][0], table_width-120)
    for i, (ymin,ymax,row) in enumerate(table):
        xmin, xmax, txt = row[0]
        w, h = xmax-xmin, ymax-ymin
        if xmin < left + 2 * th and w > tw*0.55:
            title_last_row = i
        else:
            break

    for i, (ymin,ymax,row) in enumerate(table):
        if i > title_last_row and len(row) > 2:
            table_first_row = i

    if table_first_row == -1:
        for ymin,ymax,row in table:
            if i > title_last_row and len(row) > 1:
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
    nr_col = len(table[table_first_row][2])
    for i, (ymin,ymax,row) in enumerate(table):
        if i > table_first_row:
            xmin, xmax, txt = row[0]
            if len(row) < 3 and xmin-xmax > 3 * table_width/nr_col:
                break
            if xmax-xmin > table_width*0.5 or len(txt.split()) > 8:
                break
            table_last_row = i

    return title_last_row, table_first_row, table_last_row + 1


def read_layout(layout_path):
    layout = json.load(layout_path)
    # layout = aligment(layout)

    title_ix, table_ix, note_ix = check_table_position(layout)

    t_boxes = []
    for y0, y1, row in layout[table_ix:note_ix]:
        for x0, x1, txt in row:
            t_boxes.append([x0, y0, x1-x0, y1-y0, txt])

    return layout, t_boxes, title_ix, table_ix, note_ix


def process(imgfile, debug=False):
    layout_path = imgfile.replace(".full.png", ".json")
    layout, t_boxes, title_ix, table_ix, note_ix = read_layout(open(layout_path))

    if title_ix < 0 or table_ix < 0:
        print(imgfile, title_ix, table_ix)
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
    for i in range(table_ix):
        xmin, xmax, txt = layout[i][2][0]
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

    s_boxes = []
    s_boxes_temp = segment(image[title_y1:table_y0,])
    for x,y,w,h in s_boxes_temp:
        y += title_y1
        if w > 100 and h > 100:
            s_boxes.append((x,y,w,h))

    r_boxes = struct_search(image, t_boxes, table_y0, table_y1)
    table, hl, vl = maketable(r_boxes, t_boxes)
    return table, title, note, s_boxes, r_boxes, t_boxes

def struct_search(image, t_boxes, table_y0, table_y1):
    r_boxes = []
    r_boxes_temp = segment(image[table_y0:table_y1,])
    for x,y,w,h in r_boxes_temp:
        y += table_y0
        if w > 50 and h > 50:
            r_boxes.append((x,y,w,h))
    merge_box(r_boxes, t_boxes, (3,3))
    table, hl, vl = maketable(r_boxes, t_boxes)

    rj = match_column(r_boxes, t_boxes)
    for j in range(1, len(table[0])):
        cname = table[0][j].upper()
        if cname in ("R", "R'", "R1", "R2", "R3", "R4", "X", "Y"):
            if j not in rj:
                rj.append(j)
    for j in rj:
        y_offset = hl[1][0]
        r_boxes_temp = segment(image[y_offset:table_y1, vl[j][0]:vl[j][1]])
        for x,y,w,h in r_boxes_temp:
            if w > 10 and h > 10:
                x += vl[j][0]
                y += y_offset
                r_boxes.append((x,y,w,h))
    merge_box(r_boxes, t_boxes, (3,3))
    return r_boxes

def match_column(r_boxes, t_boxes):
    rvl = scanline(r_boxes, axis=0)
    tvl = scanline(t_boxes, axis=0)
    rj = []
    for s,t in rvl:
        for i, (x0, x1) in enumerate(tvl):
            if s<=x0<=t or s<=x1<=t or (x0<=s and t<=x1):
                rj.append(i)
                break
    return rj


def merge_box(r_boxes, t_boxes, margin=1):
    valign(r_boxes)
    for i in range(len(t_boxes), 0, -1):
        drop = False
        for j in range(len(r_boxes)):
            if check_intersection(t_boxes[i-1], r_boxes[j], margin):
                xa,ya,wa,ha = r_boxes[j][:4]
                xb,yb,wb,hb = t_boxes[i-1][:4]
                x1,x2 = min(xa, xb), max(xa+wa, xb+wb)
                y1,y2 = min(ya, yb), max(ya+ha, yb+hb)
                r_boxes[j] = [x1, y1, x2-x1, y2-y1]
                drop = True
        if drop:
            del t_boxes[i-1]
    valign(r_boxes)


def valign(r_boxes):
    svl = scanline(r_boxes, axis=0)
    shl = scanline(r_boxes, axis=1)
    for k, (x,y,w,h) in enumerate(r_boxes):
        i,j = locate(y,y+h,shl), locate(x,x+w,svl)
        (x0, x1), (y0, y1) = svl[j], shl[i]
        r_boxes[k] = [x0,y0,x1-x0,y1-y0]


def maketable(r_boxes, t_boxes, shrink=0.1):
    all_boxes = []
    for x,y,w,h,t in t_boxes:
        all_boxes.append([x,y,w,h])
    for x,y,w,h in r_boxes:
        s = int(w*shrink)
        all_boxes.append([x+s, y, w-2*s, h])

    hl = scanline(all_boxes, axis=1)
    vl = scanline(all_boxes, axis=0)

    table = [[[] for j in range(len(vl))] for i in range(len(hl))]

    for c,(x,y,w,h,t) in enumerate(t_boxes):
        i = locate(y,y+h,hl)
        j = locate(x,x+w,vl)
        table[i][j].append(normalize_text(t))

    for i in range(len(table)):
        for j in range(len(table[i])):
            table[i][j] = ';'.join(table[i][j])

    for c,(x,y,w,h) in enumerate(r_boxes):
        i = locate(y,y+h,hl)
        j = locate(x,x+w,vl)
        table[i][j] = [x,y,w,h]

    return table, hl, vl
