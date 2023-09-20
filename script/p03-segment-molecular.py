import os
import re
import cv2
import json
import numpy as np
from glob import glob
from imgutil import read_image, segment
from layout import read_layout

def check_intersection(boxA, boxB, margin):
    x = max(boxA[0], boxB[0])
    y = max(boxA[1], boxB[1])
    w = min(boxA[0] + boxA[2], boxB[0] + boxB[2]) - x
    h = min(boxA[1] + boxA[3], boxB[1] + boxB[3]) - y
    marginX, marginY = margin
    return w > -marginX and h > -marginY

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


def normalize_text(txt):
    if re.match('^[0-9.]+\s+[0-9.]+[abc]?$', txt):
        a, b = txt.split()
        txt = f"{a}±{b}"
    txt = re.sub(u"[1I]C[5sS][0oO]", "IC50", txt)
    syns = {"R'": "R1"}
    return syns.get(txt, txt)


def detect_rgroup_columns(table):
    rj, header = [], table[0]
    for j, t in enumerate(header):
        if isinstance(t, str) and t.upper() in ("R", "R'", "R1", "R2", "R3", "R4", "X", "Y", "Ar"):
            rj.append(j)
    return rj


def struct_search(image, t_boxes, table_y0, table_y1):
    r_boxes = []
    r_boxes_temp = segment(image[table_y0:table_y1,])
    for x,y,w,h in r_boxes_temp:
        y += table_y0
        if w > 50 and h > 50:
            r_boxes.append((x,y,w,h))

    merge_box(r_boxes, t_boxes, (3,3))

    #table, hl, vl = maketable(r_boxes, t_boxes)
    #rj = match_column(r_boxes, t_boxes) + detect_rgroup_columns(table)
    #rj = sorted(list(set(rj)))
    #y_offset = hl[1][0] if len(hl) > 1 else table_y0
    #for j in rj:
    #    x0, x1 = vl[j][0], vl[j][1]
    #    y0, y1 = y_offset, table_y1
    #    r_boxes_temp = segment(image[y0:y1, x0:x1])
    #    for x,y,w,h in r_boxes_temp:
    #        if w > 10 and h > 10:
    #            r_boxes.append((x+x0,y+y0,w,h))
    #merge_box(r_boxes, t_boxes, (3,3))
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


def process(imgfile, debug=False):
    layout_path = imgfile.replace(".png", ".json")
    layout, t_boxes, title_ix, table_ix, note_ix = read_layout(layout_path)

    if title_ix < 0 or table_ix < 0:
        return

    image = read_image(imgfile)
    height, width = image.shape
    for i, (ymin, ymax, row) in enumerate(layout):
        if i <= title_ix or i >= table_ix:
            for xmin, xmax, txt in row:
                x0 = max(0, xmin-1)
                y0 = max(0, ymin-1)
                x1 = min(width, xmax+1)
                y1 = min(height, ymax+1)
                image[y0:y1,x0:x1] = 0

    title_y1 = layout[title_ix][1]
    table_y0 = layout[table_ix][0]
    table_y1 = layout[note_ix][0] if note_ix < len(layout) else image.shape[0]

    s_boxes, s_boxes_temp = [], segment(image[title_y1:table_y0,])
    for x,y,w,h in s_boxes_temp:
        y += title_y1
        if w > 100 and h > 100:
            s_boxes.append((x,y,w,h))

    r_boxes = struct_search(image, t_boxes, table_y0, table_y1)

    data = json.load(open(layout_path))
    data["scaffold"] = [(b, "") for b in s_boxes]
    data["rgroup"] = [(b, "") for b in r_boxes]
    json.dump(data, open(layout_path, "w"))
    return data


def render_segment(imgfile):
    data = json.load(open(imgfile.replace(".png", ".json")))
    r_boxes, s_boxes, t_boxes = data["scaffold"], data["rgroup"], data["text"]
    image = cv2.imread(imgfile)
    r, g, b = (0,0,255), (0,255,0), (255,0,0)
    for box in s_boxes:
        cv2.rectangle(image, box, r, 2)
    for box in r_boxes:
        cv2.rectangle(image, box, g, 2)
    for box, text in t_boxes:
        cv2.rectangle(image, box, b, 1)
    cv2.imwrite(imgfile.replace(".png", ".seg.png"), image)


def save_mol_image(img, prefix, shapes):
    for (x0,y0,w,h),_ in shapes:
        molfile = f"{prefix}.{x0}_{y0}_{w}_{h}.png"
        if not os.path.exists(molfile):
            mol = img[y0:y0+h,x0:x0+w,:]
            cv2.imwrite(molfile, mol)

if __name__ == "__main__":
    import sys
    fpat = sys.argv[1] if len(sys.argv) > 1 else '*'
    for imgfile in glob(f"{DOC_ROOT}/{fpat}/*.png"):
        try:
            data = process(imgfile)
            if data is None:
                print(f"{imgfile} has no tables")
            else:
                img = cv2.imread(imgfile)
                prefix = imgfile.replace(".png", "").replace("images/", "osra/")
                save_mol_image(img, prefix + ".s", data["scaffold"])
                save_mol_image(img, prefix + ".r", data["rgroup"])
                print(f"{imgfile}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"{imgfile} -> {e}", file=sys.stderr)
