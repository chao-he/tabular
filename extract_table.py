import os
import re
import sys
import json
import fitz
import numpy as np
from glob import glob
from multiprocessing import Pool
from document import parse_doc
from config import PDF_ROOT, DOC_ROOT


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


def find_table_note(page, bbox, direction=0):
    for b in page.get_text('dict', clip=bbox)["blocks"]:
        if b["type"] == 1:
            continue
        for l in b["lines"]:
            s = l["spans"][0]
            t = s["text"]
            o = fitz.Point(s["origin"])
            r = fitz.Rect(l["bbox"])
            if t == 'a':
                if direction == 0 and (r.y1 - o.y)/r.height > 0.5:
                    note = page.get_text('text', clip=fitz.Rect(bbox.x0, r.y0, bbox.x1, bbox.y1))
                    return bbox.y1 - r.y0, note
                if direction == 1 and (r.x1 - o.x)/r.width > 0.5:
                    note = page.get_text('text', clip=fitz.Rect(r.x0, bbox.y0, bbox.x1, bbox.y1))
                    return bbox.x1 - r.x0, note
    return 0, ""


def extract_tables2(source, target):
    try:
        doc = fitz.open(source)
        title, abstract, content, tables, figures, schemes, images = parse_doc(doc)

        parsed_doc = {"title": title, "abstract": abstract, "tables": [],
                      "mediabox": list(doc[0].mediabox)}
        for name, pn, bbox, hh in tables:
            direction = 0
            for b in doc[pn].get_text("dict", clip=bbox)["blocks"]:
                if b["type"] != 0:
                    continue
                l = b["lines"][0]
                if l["dir"][0] == 0:
                    direction = 1
                    break
            note_h, note = find_table_note(doc[pn], bbox, direction)
            parsed_doc["tables"].append({"title": name, "page":pn, "bbox": list(bbox), "direction": direction,
                                         "title_h": hh,  "note_h": note_h, "note": note})
        with open(f"{target}/document.json", "w") as output:
            output.write(json.dumps(parsed_doc))
        print(f"-> {target}/document.json")
    except Exception as e:
        import traceback
        print(f"-> {target}/document.json, error: ", e)


def process(source):
    target = DOC_ROOT + "/" + os.path.basename(source)[:-4]
    extract_tables2(source, target)


if __name__ == "__main__":
    with Pool(60) as p:
        p.map(process, glob(f"{PDF_ROOT}/acs.jmedchem.0c00450.pdf"))

    #for source in glob(f"{PDF_ROOT}/*.pdf"):
    #    target = DOC_ROOT + "/" + os.path.basename(source)[:-4]
    #    if not os.path.exists(target):
    #        os.makedirs(target, exist_ok=True)
    #    extract_tables2(source, target)
