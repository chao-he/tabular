import cv2
import numpy as np


def read_binary_image(imgfile):
    img = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    return 255-img if img is not None else None


def blur(image, ksize):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize,ksize))
    image = cv2.dilate(image, kernel, iterations=3)
    return cv2.erode(image, kernel, iterations=3)


def denoise(image, kshape):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kshape)
    image2 = cv2.erode(image, kernel, iterations=3)
    return cv2.dilate(image2, kernel, iterations=5)


def remove_lines(image, lpad=50, ksize=4, hsize=50):
    xl = denoise(image, (min(lpad, image.shape[1]//10), 1))
    yl = denoise(image, (1, min(lpad, image.shape[0]//10)))

    lines = cv2.addWeighted(xl, 0.5, yl, 0.5, 0)
    _, lines = cv2.threshold(lines, 128, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    lines = cv2.erode(~lines, kernel, iterations=2)
    _, lines = cv2.threshold(lines, 128, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

    return cv2.bitwise_and(image, lines)
    #image = cv2.fastNlMeansDenoising(image, h=hsize)
    #_, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    #return image


def read_image(imgfile):
    img = cv2.imread(imgfile)
    if img is None:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, gray = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    gray = remove_lines(255 - gray)
    return blur(gray, 2)


def get_bbox(img):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contours]


def segment(img, iteration=10):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))
    img = cv2.dilate(img, kernel, iterations=3)

    boxes, canvas = get_bbox(img), np.zeros_like(img)
    for i in range(iteration):
        for box in boxes:
            cv2.rectangle(canvas, box, 255, -1)
        n, boxes = len(boxes), get_bbox(canvas)
        if len(boxes) == n:
            break
    return boxes
