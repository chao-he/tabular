import sys
import cv2
import numpy as np
import paddleocr
from glob import glob


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
    try:
        blocks = []
        dt_boxes, rec_res, _ = segment_fn(img, cls=False)
        for box, (text, conf) in zip(dt_boxes, rec_res):
            tl, br = np.min(box, axis=0), np.max(box, axis=0)
            bbox = np.concatenate([tl, br], dtype=np.int32) + [0, 1, 0, -1]
            blocks.append([bbox, text])
        return aligment(blocks)
    except Exception as e:
        print(e)
    return []


def iter_imgfiles(ds):
    for root in batch:
        for imgfile in sorted(glob(root + "/*.full.png")):
            yield root, imgfile


def process_batch(batch):
    print("total = ", len(batch), file=sys.stderr)
    ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    for imgdir, imgfile in iter_imgfiles(batch):
        print(imgfile, file=sys.stderr)
        image = cv2.imread(imgfile)
        table = layout_analysis(image, ocr)
        with open(imgfile.replace(".png", ".csv"), "w") as output:
            for ymin, ymax, row in table:
                text = [" ".join(text.split()) for xmin, xmax, text in row]
                print('\t'.join(text), file=output)


def process(nproc=10):
    from multiprocessing import Pool
    from more_itertools import batched
    docs = list(glob("./tables/*"))
    ds = batched(docs, len(docs)//nproc + nproc)
    with Pool(nproc) as p:
        p.map(process_batch, ds)

if __name__ == "__main__":
    process_batch(list(sorted(glob("./tables/*"))))
