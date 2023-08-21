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



def dump_table(imgfile, blocks):
    table = aligment(blocks)
    with open(imgfile + ".csv", "w") as txt:
        for ymin, ymax, row in table:
            for xmin, xmax, text in row:
                print(" ".join(text.split()), end='\t', file=txt)
            print('', file=txt)


def process_batch(batch):
    print("total = ", len(batch), file=sys.stderr)
    ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    for root in batch:
        for imgfile in sorted(glob(root + "/*.full.png")):
            img = cv2.imread(imgfile)
            if img is None:
                continue
            print(imgfile, img.shape, file=sys.stderr)
            try:
                blocks = []
                dt_boxes, rec_res, _ = ocr(img, cls=False)
                for box, res in zip(dt_boxes, rec_res):
                    text, conf = res
                    tl, br = np.min(box, axis=0), np.max(box, axis=0)
                    bbox = np.concatenate([tl, br], dtype=np.int32) + [0, 1, 0, -1]
                    blocks.append([bbox, text])
                dump_table(imgfile, blocks)
            except Exception as e:
                print(e)


def process(nproc=10):
    from multiprocessing import Pool
    from more_itertools import batched
    docs = list(glob("./tables/*"))
    ds = batched(docs, len(docs)//nproc + nproc)
    with Pool(nproc) as p:
        p.map(process_batch, ds)

if __name__ == "__main__":
    #process(nproc=1)
    process_batch(list(sorted(glob("./tables/*"))))
