import os
import re
import sys
import json
import fitz
import numpy as np
from glob import glob
from multiprocessing import Pool
from document import parse_doc


def extract_images(page, clip):
    boxes = set()

    clip = clip + fitz.Rect(-2, -2, 2, 2)
    for img in page.get_image_info(xrefs=True):
        r = fitz.Rect(img["bbox"])
        if clip.contains(r):
            boxes.add(r.round())

    for img in page.get_drawings(extended=True):
        r = img.get('scissor', None)
        if r is None or r.is_empty or r.is_infinite:
            continue
        if clip.contains(r):
            boxes.add(r.round())
    return sorted(list(boxes), key=lambda r: (r.tl.y, r.tl.x))


def extract_tables(source, target):
    try:
        segments = []
        doc = fitz.open(source)
        title, abstract, content, tables, figures, schemes, images = parse_doc(doc)
        for ix, (name, pn, bbox, hh) in enumerate(tables):
            img = doc[pn].get_pixmap(clip=bbox, dpi=256)
            img.save(f"{target}/table.{ix+1:03d}.full.png")

            #if hh < 40:
            #    bbox.y0 += hh

            #for blk in doc[pn].get_text('dict', clip=bbox)["blocks"]:
            #    for line in blk.get("lines", []):
            #        for span in line["spans"]:
            #            if span["origin"][0]-bbox.x0 < 3 and span["text"] == 'a':
            #                bbox.y1 = line["bbox"][1]
            #                break

            #img = doc[pn].get_pixmap(clip=bbox, dpi=256)
            #img.save(f"{target}-table.{ix+1}.data.png")

        # with open(f"{target}.json", "w") as output:
        #     output.write(json.dumps(segments))
        print(f"{source} -> {target}")
    except Exception as e:
        print(e, source, target)
    return segments


if __name__ == "__main__":
    #for source in glob("./papers/*.pdf"):
    for source in open("discovery.txt"):
        source = source.strip()
        target = "tables/" + os.path.basename(source)[:-4]
        if os.path.exists(target):
            continue
        os.makedirs(target, exist_ok=True)
        extract_tables(source, target)
