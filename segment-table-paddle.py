import sys
import cv2
import json
import numpy as np
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


def check_header(table):
    first_column = []
    for ix, (ymin, ymax, row) in enumerate(table):
        if len(row) > 3:
            xmin, xmax, text = row[0]
            word = text.lower().split()[0].strip(".").strip("#")
            first_column.append([ix, word])
    for ix, word in first_column:
        if word in ("compound", "entry", "structure", "fragment"):
            return ix
    for ix, word in first_column:
        if word[:4] in ("comp", "cmpd", "cpd"):
            return ix
    for ix, word in first_column:
        if word in ("id", "no",  "ex"):
            return ix
    return 0


def layout_analysis(img, segment_fn):
    blocks = []
    dt_boxes, rec_res, _ = segment_fn(img, cls=False)
    for box, (text, conf) in zip(dt_boxes, rec_res):
        tl, br = np.min(box, axis=0), np.max(box, axis=0)
        (top, left), (bottom, right) = tl, br
        bbox = np.array(list(map(int, [top, left, bottom, right])))
        blocks.append([bbox, text])
    return aligment(blocks)


def iter_imgfiles(ds):
    for root in ds:
        for imgfile in sorted(glob(root + "/*.full.png")):
            yield root, imgfile


def process(batch):
    print("total = ", len(batch), file=sys.stderr)
    ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    for imgdir, imgfile in iter_imgfiles(batch):
        try:
            image = cv2.imread(imgfile)
            table = layout_analysis(image, ocr)
            json_path = imgfile.replace(".full.png", ".json")
            with open(json_path, "w") as output:
                json.dump(table, output)
            print(json_path, file=sys.stderr)
        except Exception as e:
            print(imgfile, e, file=sys.stderr)


def process_mp(nproc=2):
    from multiprocessing import Pool
    from more_itertools import batched

    docs = sorted(list(glob(DOC_ROOT + "/*")))
    batch = batched(docs, len(docs)//nproc + nproc)
    with Pool(nproc) as p:
        p.map(process, batch)

if __name__ == "__main__":
    #process_mp(2)
    docs = sorted(list(glob(DOC_ROOT + "/*")))
    process(docs)
