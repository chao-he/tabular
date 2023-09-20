import os
import sys
import cv2
import json
import math
import numpy as np
import traceback
import paddleocr
from glob import glob
from config import DOC_ROOT


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
    width = np.max([b[0][2] for b in boxes]) if boxes else 0
    height = np.max([b[0][3] for b in boxes]) if boxes else 0

    flags = np.zeros(height)
    for b,t in boxes:
        flags[b[1]:b[3]] = 1

    rows = []
    for ymin, ymax in nonzero(flags):
        cells, mask = [], np.zeros(width)
        for (x,y,w,h),text in boxes:
            if y>=ymin and h<=ymax:
                mask[x:w] = 1
                cells.append([x,y,w,h,text])
        columns = []
        for xmin, xmax in nonzero(mask):
            column = [(y, t) for x,y,w,h,t in cells if x>=xmin and w<=xmax]
            text = " ".join([t for y, t in sorted(column)])
            columns.append([xmin, xmax, text])
        rows.append([ymin, ymax, columns])
    return rows


def layout_analysis(img, segment_fn):
    blocks = []
    dt_boxes, rec_res, _ = segment_fn(img, cls=False)
    for box, (text, conf) in zip(dt_boxes, rec_res):
        (left, top), (right, bottom) = np.min(box, axis=0), np.max(box, axis=0)
        bbox = math.floor(left), math.floor(top), math.ceil(right-left), math.ceil(bottom-top)
        blocks.append([bbox, text])
    return blocks


def iter_imgfiles(ds):
    for root in ds:
        for imgfile in sorted(glob(root + "/*.table.*.png")):
            yield root, imgfile


def process(batch):
    print("total = ", len(batch), file=sys.stderr)
    ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang='latin', det_box_type='poly', show_log=False)

    failures = []
    for _, imgfile in iter_imgfiles(batch):
        json_path = imgfile.replace(".png", ".json")
        if not os.path.exists(json_path):
            image = cv2.imread(imgfile)
            try:
                table = layout_analysis(image, ocr)
                with open(json_path, "w") as output:
                    json.dump({"text": table}, output)
                print(json_path, file=sys.stderr)
            except Exception as e:
                print(json_path, e, file=sys.stderr)
                failures.append([json_path, image])
                traceback.print_exc()

    ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang='latin', det_box_type='quad', show_log=False)
    for json_path, image in failures:
        try:
            table = layout_analysis(image, ocr)
            with open(json_path, "w") as output:
                json.dump({"text": table}, output)
            print(json_path, file=sys.stderr)
        except Exception as e:
            traceback.print_exc()

if __name__ == "__main__":
    import sys
    fpat = sys.argv[1] if len(sys.argv) > 1 else '*'
    docs = sorted(list(glob(f"{DOC_ROOT}/{fpat}")))
    print(docs)
    process(docs)
