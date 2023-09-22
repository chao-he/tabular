import re
import os
import sys
import json
import numpy as np
import cv2
from glob import glob
from config import DOC_ROOT

def parse_table_id(title):
    valid_table_pattern = r'^Table\s+([0-9]+)(?:\.|\s+Continued|$)'
    words, sentences = title.split(), title.split(".")
    m = re.match(valid_table_pattern, title, re.I)
    if m and (len(words) <= 50 or len(sentences) <= 3):
        tix = m.group(1)
        if title.lower().endswith('continued'):
            tix = tix + ".cont"
        return tix

def pt2pix(x):
    return int(round(float(x/72*256)))


def render_segment(imgfile, th, nh):
    data = json.load(open(imgfile.replace(".png", ".json")))
    if "scaffold" not in data:
        return

    r_boxes, s_boxes, t_boxes = data["scaffold"], data["rgroup"], data["text"]

    image = cv2.imread(imgfile)
    r, g, b = (0,255,0), (0,0,255), (255,0,0)
    for box,_ in s_boxes:
        cv2.rectangle(image, box, r, 2)
    for box,_ in r_boxes:
        cv2.rectangle(image, box, g, 2)
    for box,_ in t_boxes:
        cv2.rectangle(image, box, b, 1)

    h,w,c = image.shape
    if th > 0:
        s = image[:th,:,:]
        s[s==255] = 200
    if nh > 0:
        s = image[nh:,:,:]
        s[s==255] = 200
    cv2.imwrite(imgfile.replace(".png", ".seg.png"), image)



if __name__ == "__main__":
    doi = sys.argv[1] if len(sys.argv) > 1 else "*"
    for root in glob(f"{DOC_ROOT}/{doi}"):
        doi = os.path.basename(root)
        doc = json.load(open(f"{DOC_ROOT}/{doi}/document.json"))

        nmap = {}
        for t in doc["tables"]:
            tix = parse_table_id(t["title"])
            if tix:
                x0, y0, x1, y1 = t["bbox"]
                nmap[tix] = (pt2pix(y1-y0-t["note_h"]), pt2pix(t["title_h"]))

        for imgfile in glob(f"{DOC_ROOT}/{doi}/*.table.*.png"):
            if imgfile.count('.seg.') > 0:
                continue
            items = imgfile.split(".table.")[-1].split(".")

            tix = items[0]
            if items[1] == "cont":
                tix = items[0] + ".cont"
            nh, th = nmap.get(tix, (0,0))
            render_segment(imgfile, th, nh)
            print(imgfile, tix)
