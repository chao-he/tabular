import sys
import json
import cv2


def render_segment(imgfile):
    data = json.load(open(imgfile.replace(".png", ".json")))
    r_boxes, s_boxes, t_boxes = data["scaffold"], data["rgroup"], data["text"]
    image = cv2.imread(imgfile)
    r, g, b = (0,0,255), (0,255,0), (255,0,0)
    for box,_ in s_boxes:
        cv2.rectangle(image, box, r, 2)
    for box,_ in r_boxes:
        cv2.rectangle(image, box, g, 2)
    for box,_ in t_boxes:
        cv2.rectangle(image, box, b, 1)
    cv2.imwrite(imgfile.replace(".png", ".seg.png"), image)


if __name__ == "__main__":
    render_segment(sys.argv[1])
