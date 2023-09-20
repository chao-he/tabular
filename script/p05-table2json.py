import os
import re
import cv2
import sys
import json
from rdkit import Chem
from glob import glob
from segment import process
from collections import defaultdict
from config import DOC_ROOT


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



def normalize_smiles(smiles):
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    except Exception as e:
        return None


def parse_table(doc):
    tables = []
    for imgfile in sorted(glob(f"{doc}/*.full.png")):
        try:
            result = process(imgfile)
            if result is None:
                print(imgfile, ' parse fail')
                continue

            table, title, note, s_boxes, r_boxes, t_boxes = result

            image = cv2.imread(imgfile, 1)
            for x,y,w,h in s_boxes:
                fpath = makepath(imgfile, f".s.{x}_{y}_{w}_{h}.png")
                cv2.imwrite(fpath, image[y:y+h,x:x+w])
            for x,y,w,h in r_boxes:
                fpath = makepath(imgfile, f".r.{x}_{y}_{w}_{h}.png")
                cv2.imwrite(fpath, image[y:y+h,x:x+w])

            image = cv2.imread(imgfile)
            r, g, b = (0,0,255), (0,255,0), (255,0,0)
            for x,y,w,h in s_boxes:
                cv2.rectangle(image, (x,y,w,h), r, 2)
            for x,y,w,h in r_boxes:
                cv2.rectangle(image, (x,y,w,h), g, 2)
            for x,y,w,h,t in t_boxes:
                cv2.rectangle(image, (x,y,w,h), b, 1)
            cv2.imwrite(makepath(imgfile, ".dbg.png"), image)

            tables.append({
                "title": title, "note": note, "content": table,
                "source": makepath(imgfile, ".dbg.png", True),
            })

            nr_row, nr_col = len(table), len(table[0])
            for i in range(nr_row):
                for j in range(nr_col):
                    if isinstance(table[i][j], list):
                        x,y,w,h = table[i][j]
                        table[i][j] = makepath(imgfile, f".r.{x}_{y}_{w}_{h}.png", True)
            print(imgfile, nr_row, nr_col, "done")
        except Exception as e:
            import traceback
            traceback.print_exc()
    return tables

def makepath(imgfile, suffix, only_basename=False):
    if only_basename:
        return os.path.basename(imgfile.replace(".full.png", suffix))
    return os.path.abspath(imgfile.replace(".full.png", suffix))


if __name__ == "__main__":
    for doc in glob(f"{DOC_ROOT}/*"):
        doi = os.path.basename(doc)
        tables = parse_table(doc)
        with open(f"data/{doi}.json", "w") as output:
            json.dump(tables, output)
