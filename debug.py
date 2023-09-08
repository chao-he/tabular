import sys
import fitz

for pdf in sys.argv[1:]:
    print(pdf)
    doc = fitz.open(pdf)

    for p in doc:
        for b in p.get_text('dict')["blocks"]:
            if b["type"] == 1:
                continue
            for l in b["lines"]:
                lb = fitz.Rect(l["bbox"])
                text = [s["text"] for s in l["spans"]]

                has_s = False
                for s in l["spans"]:
                    o = fitz.Point(s["origin"])
                    y = (lb.y1 - o.y)/lb.height
                    if y > 0.5:
                        has_s = True
                if has_s:
                    print(text)
    print()
