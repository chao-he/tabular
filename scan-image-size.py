from glob import glob
import os
import cv2

for imgf in glob("../papers/tables/*/*.full.png"):
    img = cv2.imread(imgf)
    h, w, c = img.shape
    print(w,h,imgf)

