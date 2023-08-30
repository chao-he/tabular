import os
import re
import cv2
import sys
import json
import jinja2
from rdkit import Chem
from glob import glob
from segment import process
from collections import defaultdict
from config import DOC_ROOT


def normalize_smiles(smiles):
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    except Exception as e:
        return None

def gen_html(path, tables):
    tpl = jinja2.Template(open("html/template.html").read())
    html = tpl.render(tables=tables)
    with open(path, "w") as output:
        output.write(html)

def doc_to_html(doc):
    doi = doc.split("/")[-1]
    tables = []
    for imgfile in sorted(glob(f"{doc}/*.full.png")):
        try:
            table, title, note, s_boxes, r_boxes, t_boxes = process(imgfile)

            image = cv2.imread(imgfile, 1)
            for x,y,w,h in s_boxes:
                fpath = imgfile.replace(".full.png", f".s.{x}_{y}_{w}_{h}.png")
                cv2.imwrite(fpath, image[y:y+h,x:x+w])
            for x,y,w,h in r_boxes:
                fpath = imgfile.replace(".full.png", f".r.{x}_{y}_{w}_{h}.png")
                cv2.imwrite(fpath, image[y:y+h,x:x+w])

            image = cv2.imread(imgfile)
            r, g, b = (0,0,255), (0,255,0), (255,0,0)
            for x,y,w,h in s_boxes:
                cv2.rectangle(image, (x,y,w,h), r, 2)
            for x,y,w,h in r_boxes:
                cv2.rectangle(image, (x,y,w,h), g, 2)
            for x,y,w,h,t in t_boxes:
                cv2.rectangle(image, (x,y,w,h), b, 1)
            cv2.imwrite(imgfile.replace(".full.png", ".dbg.png"), image)

            tables.append({
                "source": os.path.basename(imgfile.replace(".full.png", ".dbg.png")),
                "title": title,
                "content": table,
                "note": note
            })

            for i in range(len(table)):
                for j in range(len(table[i])):
                    if isinstance(table[i][j], list):
                        x,y,w,h = table[i][j]
                        src = os.path.basename(imgfile.replace(".full.png", f".r.{x}_{y}_{w}_{h}.png"))
                        table[i][j] = f"<img width=\"{w//3}\" src=\"{src}\"/>"
            print(imgfile, "done")
        except Exception as e:
            import traceback
            traceback.print_exc()
    gen_html(f"{doc}/index.html", tables)


def extract_compounds(doc):
    cpd_path = f"{doc}/compounds.txt"
    if not os.path.exists(cpd_path):
        return

    doi = os.path.basename(doc)

    compounds = {}
    for line in open(cpd_path):
        cid, smiles = line.strip().split("\t")
        smiles = normalize_smiles(smiles)
        if smiles:
            compounds[cid] = {"smiles": smiles, "source": doi}

    for imgfile in sorted(glob(f"{doc}/*.full.png")):
        try:
            table, title, note, _ = process(imgfile)
            head = table[0]
            for row in table[1:]:
                if not row:
                    continue
                attrs = compounds.get(row[0], None)
                if not attrs:
                    continue
                for j in range(1, len(row)):
                    if not head[j] or head[j] in ('R', 'X', 'R1', 'R2', 'R3', 'R4', 'Ar'):
                        continue
                    attrs[head[j]] = row[j]
            print(imgfile, "done")
        except Exception as e:
            print(imgfile, e)
    return compounds


if __name__ == "__main__":
    for doc in glob("./tables/*"):
        doc_to_html(doc)
    #output = open("33k-dataset.jsonl", "w")
    #for doc in glob(DOC_ROOT + "/*"):
    #    compounds = extract_compounds(doc)
    #    if compounds:
    #        print(json.dumps(compounds), file=output)
    #output.close()
