import os
import re
import sys
import json
from collections import defaultdict
from glob import glob
from config import DOC_ROOT
from layout import read_layout

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
    r = []
    for s,t in segs:
        if s < t:
            if r:
                r[-1] = (r[-1]+s)//2
                r.append(t)
            else:
                r += [t]
    return r


def locate(xmin, xmax, lines):
    x = (xmin + xmax)/2
    for i, s in enumerate(lines):
        if x < s:
            return i
    return len(lines) - 1

def maketable(r_boxes, t_boxes, scale=1.0):
    all_boxes = []
    for x,y,w,h,t in t_boxes:
        x = x + w*(1-scale)/2
        all_boxes.append([x, y, w*scale, h])

    for x,y,w,h,t in r_boxes:
        x = x + w*(1-scale)/2
        all_boxes.append([x, y, w*scale, h])

    hl = scanline(all_boxes, axis=1)
    vl = scanline(all_boxes, axis=0)

    table = [[[] for j in range(len(vl))] for i in range(len(hl))]

    for c,(x,y,w,h,t) in enumerate(t_boxes):
        i = locate(y,y+h,hl)
        j = locate(x,x+w,vl)
        table[i][j].append(t)
    return table, hl, vl


if __name__ == "__main__":
    data = json.load(open(sys.argv[1]))
    r_boxes = []
    for (x,y,w,h),t in data.get("scaffold", []):
        r_boxes.append([x,y,w,h,t])
    _,t_boxes,_,_,_ = read_layout(data["text"])
    table, hl, vl = maketable(r_boxes, t_boxes, 1.0)
    print(len(hl), len(vl))
    for row in table:
        txt = '\t'.join([';'.join(c) if c else '-' for c in row])
        print(txt)
