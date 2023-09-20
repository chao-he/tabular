import os
import re
import json
import fitz
import cv2
import numpy as np
from glob import glob


def get_image(page, direction, bbox, dpi=256):
    pix = page.get_pixmap(clip=bbox, dpi=dpi, matrix=fitz.Matrix(2,2))
    img = np.frombuffer(buffer=pix.samples_mv, dtype=np.uint8)\
        .reshape((pix.height, pix.width, -1))

    inc2pix = pix.width/bbox.width

    bottom, right = img.shape

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lo, hi = np.array([156,  43,  46]), np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lo, hi)
    mask[mask>0] = 1
    for i in range(60, mask.shape[0]):
        if np.sum(mask[i,:]) > min(800, img.shape[1]*0.75):
            bottom = i
            break

    if bbox.x0 < 300 and bbox.width > 400:
        l,r = round((310 - bbox.x0) * inc2pix), round((320 - bbox.x0) * inc2pix)
        vline = cv2.cvtColor(img[:bottom,l:r], cv2.COLOR_BGR2GRAY)
        if np.size(vline) == np.sum(vline==255):
            right = l

    img = img[:bottom,:right,:]
    if direction == 1:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def iter_docs(root):
    for pdf in glob(f"{root}/papers/acs.jmedchem.1c01403.pdf"):
        doi = os.path.basename(pdf)[:-4]
        yield doi, pdf, f"{root}/output/{doi}/document.json"


if __name__ == "__main__":
    valid_table_pattern = r'^Table\s+([0-9]+)(?:\.|\s+Continued|$)'

    root = os.path.expanduser("~/papers/z20230901")
    for doi, pdf, meta in iter_docs(root):
        doc, data = fitz.open(pdf), json.load(open(meta))
        os.makedirs(f"images/{doi}", exist_ok=True)
        for tbl in data["tables"]:
            title, pn, bbox, direction = tbl["title"], tbl["page"], tbl["bbox"], tbl["direction"]
            words, sentences = title.split(), title.split(".")
            m = re.match(valid_table_pattern, title, re.I)
            if m and (len(words) <= 50 or len(sentences) <= 3):
                tix = m.group(1)
                if title.lower().endswith('continued'):
                    tix = tix + ".cont"
                imgpath = f"images/{doi}/{doi}.table.{tix}.png"

                bbox = fitz.Rect(bbox)
                bbox.y1 = min(770, bbox.y1)
                print(imgpath, '\t', title, bbox)
                img = get_image(doc[pn], direction, bbox)
                cv2.imwrite(imgpath, img)
