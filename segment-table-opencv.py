import cv2
import numpy as np
import pytesseract
from functools import cmp_to_key
from multiprocessing import Pool
from more_itertools import batched
from glob import glob


def cmp_couter(a, b):
    xa,ya,wa,ha = a
    xb,yb,wb,hb = b
    if ya+ha < yb:
        return -1
    elif ya > yb+hb:
        return 1
    else:
        return -1 if xa < xb else 1

def read_binary_image(imgfile):
    img = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)
    #img[img<255] = 0
    thresh, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    return 255-img

def detect_line(image, shape):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, shape)
    image2 = cv2.erode(image, kernel, iterations=3)
    return cv2.dilate(image2, kernel, iterations=5)

def remove_lines(image, lpad=50, ksize=4, hsize=50):
    xl = detect_line(image, (min(lpad, image.shape[1]//10), 1))
    yl = detect_line(image, (1, min(lpad, image.shape[0]//10)))

    lines = cv2.addWeighted(xl, 0.5, yl, 0.5, 0)
    _, lines = cv2.threshold(lines, 128, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    lines = cv2.erode(~lines, kernel, iterations=2)
    _, lines = cv2.threshold(lines, 128, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

    image = cv2.bitwise_and(image, lines)
    image = cv2.fastNlMeansDenoising(image, h=hsize)
    _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    return image

def segment_line(img, mb=6):
    position, border = [], []
    for ix, row in enumerate(img):
        if np.sum(row) == 0:
            if position and ix == position[-1] + border[-1]:
                border[-1] += 1
            else:
                position.append(ix)
                border.append(1)
    lines = [0]
    for p,b in zip(position, border):
        if p > 0 and b >= mb and p + b < img.shape[0]:
            lines.append(p + b//2)
    lines.append(img.shape[0])
    return lines

def segment_table(img, ksize=8):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize,1))
    img = cv2.dilate(img, kernel, iterations=3)
    img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]

    xl = segment_line(img)
    yl = segment_line(img.transpose())
    grid = np.zeros_like(img)
    for ix in xl:
        grid[ix-2:ix+2, :] = 255
    for ix in yl:
        grid[:, ix-2:ix+2] = 255
    return xl, yl, grid

def extract_shapes(gray, minlen=100, ksize=4):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize,ksize))
    gray = cv2.dilate(gray, kernel, iterations=3)
    gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    shapes = []
    for i, contour in enumerate(contours):
        #area = cv2.contourArea(contour, True)
        arclen = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, max(10, 0.01*arclen), True)
        if arclen >= minlen:
            shapes.append(approx)
    return shapes

def draw_shapes(image, shapes):
    canvas = np.zeros_like(image)
    cv2.drawContours(canvas, shapes, -1, 255, 5)
    for shape in shapes:
        x,y,w,h = cv2.boundingRect(shape)
        cv2.rectangle(canvas, (x,y), (x+w,y+h), 255, 5)
    return cv2.threshold(canvas, 127, 255, cv2.THRESH_BINARY)[1]

def scaffold(image):
    xl, yl, grid = segment_table(image)
    for i in range(min(len(xl)-1, 3)):
        gray = search_scaffold(image[xl[i]:xl[i+1],:], minlen=10, ksize=4)
        if np.sum(gray) > 0:
            return xl[i+1]
    return None

def draw_bbox(img, iteration=3):
    num_cnts, canvas = 0, np.zeros_like(img)
    for i in range(iteration):
        target = img if i == 0 else canvas
        cuts = cv2.findContours(target, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        if len(cuts) == num_cnts:
            break
        for cnt in cuts:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(canvas, (x-1,y), (x+w+1,y+h), 255, -1)
        num_cnts = len(cuts)
    return cuts


def segment_(boxes, width, height):
    flags = np.zeros(height)
    for x,y,w,h in boxes:
        flags[y:y+h] = 1
    i, hl = 0, []
    while i < height:
        while i < height and flags[i] == 0:
            i += 1
        s = i
        while i < height and flags[i] == 1:
            i += 1
        hl.append([s, i])

    rows = [[] for i in range(len(hl))]
    for x,y,w,h in boxes:
        for i,(s,t) in enumerate(hl):
            if y>=s and y+h<=t:
                rows[i].append([x,y,w,h])
                break
    return [sorted(row) for row in rows]

def process_img(imgfile):
    img = read_binary_image(imgfile)
    img = remove_lines(img)
    shapes = draw_bbox(img, 100)
    boxes = [cv2.boundingRect(c) for c in shapes]
    return segment_(boxes, img.shape[1], img.shape[0])

def process_doc(root):
    for imgfile in sorted(glob(f"{root}/*.full.png")):
        try:
            rows = process_img(imgfile)
            with open(imgfile + ".csv", "w") as txt:
                for row in rows:
                    print("\t".join(row), file=txt)
            print(imgfile)
        except Exception as e:
            print(e)
            pass
    print(f"Done, {root}")

def process():
    from multiprocessing import Pool
    with Pool(20) as p:
        p.map(process_doc, sorted(glob("./tables/*")))

if __name__ == "__main__":
    imgfile = 'tables/acsinfecdis.0c00068/table.001.full.png'
    imgfile = 'tables/acs.jmedchem.1c00313/table.002.full.png'
    imgfile = 'tables/acs.jmedchem.1c00166/table.002.full.png'
    img = read_binary_image(imgfile)
    table = process_img(imgfile)
    for row in table:
        for x,y,w,h in row:
            text = pytesseract.image_to_string(img[y:y+h,x:x+w], lang='eng')
            print(" ".join(text.split()), end='\t')
        print("----")
