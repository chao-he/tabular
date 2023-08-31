import os
import cv2
import sys
import json
from rdkit import Chem
from glob import glob
from segment import process
from config import DOC_ROOT


def extract_smiles(doc):
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
        except Exception as e:
            print(imgfile, e)


if __name__ == "__main__":
    for doc in glob("./tables/*"):
        extract_smiles(doc)
