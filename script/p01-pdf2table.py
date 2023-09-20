import os
import re
import sys
import cv2
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


def check_direction(page, bbox):
    for b in page.get_text("dict", clip=bbox)["blocks"]:
        if b["type"] == 0 and b["lines"][0]["dir"][0] == 0:
            return 1
    return 0


def get_image(page, direction, bbox, dpi=256):
    pix = page.get_pixmap(clip=bbox, dpi=dpi, matrix=fitz.Matrix(2,2))
    img = np.frombuffer(buffer=pix.samples_mv, dtype=np.uint8)\
        .reshape((pix.height, pix.width, -1))

    pt2pix = pix.width/bbox.width

    bottom, right, channel = img.shape

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lo, hi = np.array([156,  43,  46]), np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lo, hi)
    mask[mask>0] = 1
    for i in range(60, mask.shape[0]):
        p, = np.nonzero(mask[i])
        if len(p) > min(800, img.shape[1]*0.75):
            if p[0] < 24:
                bottom = i
                break

    if bbox.x0 < 300 and bbox.width > 400:
        x = page.mediabox.width / 2
        l,r = round((x - bbox.x0 - 5) * pt2pix), round((x - bbox.x0 + 5) * pt2pix)
        vline = cv2.cvtColor(img[:bottom,l:r,:], cv2.COLOR_BGR2GRAY)
        print(x, l, r, np.size(vline), vline.shape, np.sum(vline==255), bbox.y1)
        if np.size(vline) == np.sum(vline==255):
            right = l
    bbox.x1 = bbox.x0 + right/pt2pix
    bbox.y1 = bbox.y0 + bottom/pt2pix
    return img[:bottom,:right,:], bbox


def extract_tables2(source, target):
    valid_table_pattern = r'^Table\s+([0-9]+)(?:\.|\s+Continued|$)'
    try:
        doi, doc = os.path.basename(source)[:-4], fitz.open(source)
        title, abstract, content, tables, figures, schemes, images = parse_doc(doc)
        parsed_doc = {
            "title": title,
            "abstract": abstract,
            "tables": [],
            "mediabox": list(doc[0].mediabox),
            "doi": doi
        }

        for title, pn, bbox, hh in tables:
            words, sentences = title.split(), title.split(".")
            m = re.match(valid_table_pattern, title, re.I)
            if m and (len(words) <= 50 or len(sentences) <= 3):
                tix = m.group(1)
                if title.lower().endswith('continued'):
                    tix = tix + ".cont"
                bbox.y1 = min(760, bbox.y1)
                direction = check_direction(doc[pn], bbox)
                image, bbox = get_image(doc[pn], direction, bbox)
                note_h, note = find_table_note(doc[pn], bbox, direction)
                if direction == 1:
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

                imgpath = f"images/{doi}/{doi}.table.{tix}.png"
                parsed_doc["tables"].append({
                    "index": tix,
                    "title": title,
                    "page": pn,
                    "bbox": list(bbox),
                    "direction": direction,
                    "title_h": hh,
                    "note_h": note_h,
                    "note": note,
                    "image": imgpath
                })
                cv2.imwrite(imgpath, image)
            print(title, '\n', '\t', bbox)
        with open(f"{target}/document.json", "w") as output:
            output.write(json.dumps(parsed_doc))
        print(f"-> {target}/document.json")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"-> {target}.document.json, error: ", e)


if __name__ == "__main__":
    import sys
    fpat = sys.argv[1] if len(sys.argv) > 1 else '*'
    for source in glob(f"{PDF_ROOT}/{fpat}.pdf"):
        target = "{DOC_ROOT}/" + os.path.basename(source)[:-4]
        if not os.path.exists(target):
            os.makedirs(target, exist_ok=True)
        extract_tables2(source, target)
